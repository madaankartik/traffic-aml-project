"""Phase 2 deep-learning models (pure PyTorch, no graph library needed).

All models share one skeleton so that ablations change exactly one thing:

    x (B, 12, N, 7) --input projection + sensor embedding-->
    temporal encoder: one GRU shared by all sensors, run over the 12 steps of each sensor
        -> h_i  (each sensor's own last hour summarised in a vector)
    spatial block: K message-passing layers over the road graph
        'none' : K position-wise MLP layers (same depth, no neighbour information)  -> "GRU (no graph)"
        'gcn'  : bidirectional diffusion convolution with the FIXED adjacency     -> "DCRNN-lite"
        'gat'  : graph attention; separate attention for upstream and downstream
                 neighbours (directed) or one shared set (undirected)            -> "ST-GAT"
    head: MLP -> 3 horizons, in standard units; the caller turns it into mph:
        pred_mph = base_mph + z * std_sensor
    where base is the sensor mean (pure DL) or the XGBoost forecast (hybrid).

"Encode then propagate": every sensor first summarises its own hour, then the K layers move
those summaries K hops along the road graph.  K hops of 1-3 km each is what lets the network see
congestion that is still several sensors away -- the thing one-hop Phase 1 features could not.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn


def _self_loops(mask: np.ndarray) -> np.ndarray:
    m = mask.copy()
    np.fill_diagonal(m, True)
    return m


def relation_masks(adjacency: np.ndarray, directed: bool = True) -> list[np.ndarray]:
    """Boolean receive-masks M[i, j] = 'sensor i receives a message from j'.

    A[i, j] > 0 means j is reachable *from* i, i.e. j is downstream of i.
    directed  -> [downstream relation  M[i,j] = A[i,j] > 0,
                  upstream relation    M[i,j] = A[j,i] > 0]
    undirected-> [M = (A + A^T) > 0]
    """
    A = adjacency > 0
    np.fill_diagonal(A, False)
    if directed:
        return [_self_loops(A), _self_loops(A.T)]
    return [_self_loops(A | A.T)]


class EdgeGATLayer(nn.Module):
    """Graph attention (Velickovic et al., 2018) computed over the edge list (sparse), several heads.

    For every edge j -> i allowed by the relation mask:
        e_ij = LeakyReLU(a_dst . W h_i + a_src . W h_j) + beta * w_ij      (w_ij = road-graph weight)
        alpha_ij = softmax over the neighbours j of i
        h_i' = concat_heads( sum_j alpha_ij W h_j )
    Working on the ~1.7k edges instead of all 207 x 207 pairs makes it ~25x cheaper.
    """

    def __init__(self, dim: int, out_dim: int, heads: int, mask: np.ndarray, weight: np.ndarray,
                 attn_dropout: float = 0.1):
        super().__init__()
        assert out_dim % heads == 0
        self.heads, self.dh = heads, out_dim // heads
        self.W = nn.Linear(dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.dh))
        self.a_dst = nn.Parameter(torch.empty(heads, self.dh))
        self.beta = nn.Parameter(torch.zeros(heads))
        nn.init.xavier_uniform_(self.a_src); nn.init.xavier_uniform_(self.a_dst)
        dst, src = np.nonzero(mask)                           # i receives from j
        self.register_buffer("dst", torch.from_numpy(dst).long())
        self.register_buffer("src", torch.from_numpy(src).long())
        self.register_buffer("w", torch.from_numpy(weight[dst, src].astype(np.float32)))
        self.n = mask.shape[0]
        self.drop = nn.Dropout(attn_dropout)
        self.last_alpha = None          # (heads, E) batch-mean attention per edge, for analysis

    def forward(self, h):                                     # h: (B, N, dim)
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.heads, self.dh)
        s_src = (Wh * self.a_src).sum(-1)                     # (B, N, heads)
        s_dst = (Wh * self.a_dst).sum(-1)
        e = Fnn.leaky_relu(s_dst[:, self.dst] + s_src[:, self.src], 0.2) + self.beta * self.w.view(1, -1, 1)
        idx = self.dst.view(1, -1, 1).expand(B, -1, self.heads)
        emax = torch.full((B, N, self.heads), -1e9, device=h.device).scatter_reduce(1, idx, e, "amax", include_self=True)
        ex = torch.exp(e - emax.gather(1, idx))
        den = torch.zeros(B, N, self.heads, device=h.device).index_add_(1, self.dst, ex)
        alpha = ex / den.gather(1, idx).clamp_min(1e-12)      # (B, E, heads)
        if not self.training:
            self.last_alpha_full = alpha.detach()          # (B, E, heads)
            self.last_alpha = alpha.mean(0).T.detach()
        alpha = self.drop(alpha)
        msg = alpha.unsqueeze(-1) * Wh[:, self.src]            # (B, E, heads, dh)
        out = torch.zeros(B, N, self.heads, self.dh, device=h.device).index_add_(1, self.dst, msg)
        return out.reshape(B, N, self.heads * self.dh)


class GATBlock(nn.Module):
    def __init__(self, dim, adjacency, directed=True, heads=4, dropout=0.1):
        super().__init__()
        masks = relation_masks(adjacency, directed)
        wt = adjacency.copy(); np.fill_diagonal(wt, 1.0)
        weights = [wt, wt.T] if directed else [np.maximum(wt, wt.T)]
        h_per = max(heads // len(masks), 1)
        self.rel = nn.ModuleList([EdgeGATLayer(dim, dim // len(masks), h_per, m, w, dropout)
                                  for m, w in zip(masks, weights)])
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        msg = torch.cat([r(h) for r in self.rel], dim=-1)
        return self.norm(h + self.drop(Fnn.elu(msg)))


class DiffusionBlock(nn.Module):
    """Bidirectional random-walk diffusion step with the FIXED adjacency (DCRNN-style, K=1)."""

    def __init__(self, dim, adjacency, dropout=0.1):
        super().__init__()
        A = adjacency.astype(np.float32).copy(); np.fill_diagonal(A, 0.0)
        def rw(M):
            d = M.sum(1, keepdims=True); d[d == 0] = 1.0
            return M / d
        self.register_buffer("Pf", torch.from_numpy(rw(A)))      # from downstream
        self.register_buffer("Pb", torch.from_numpy(rw(A.T)))    # from upstream
        self.W0, self.W1, self.W2 = (nn.Linear(dim, dim) for _ in range(3))
        self.norm = nn.LayerNorm(dim); self.drop = nn.Dropout(dropout)

    def forward(self, h):
        msg = self.W0(h) + self.W1(torch.einsum("ij,bjd->bid", self.Pf, h)) + self.W2(torch.einsum("ij,bjd->bid", self.Pb, h))
        return self.norm(h + self.drop(Fnn.elu(msg)))


class MLPBlock(nn.Module):
    """Same depth as a graph layer but each sensor only sees itself (the 'no graph' control)."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(dim, dim); self.norm = nn.LayerNorm(dim); self.drop = nn.Dropout(dropout)

    def forward(self, h):
        return self.norm(h + self.drop(Fnn.elu(self.lin(h))))


class STNet(nn.Module):
    def __init__(self, n_nodes: int, adjacency: np.ndarray, spatial: str = "gat", n_layers: int = 2,
                 directed: bool = True, in_dim: int = 7, hidden: int = 64, emb_dim: int = 16,
                 heads: int = 4, dropout: float = 0.1, n_horizons: int = 3):
        super().__init__()
        self.spatial = spatial
        self.emb = nn.Embedding(n_nodes, emb_dim)
        self.inp = nn.Linear(in_dim + emb_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        if spatial == "gat":
            blocks = [GATBlock(hidden, adjacency, directed, heads, dropout) for _ in range(n_layers)]
        elif spatial == "gcn":
            blocks = [DiffusionBlock(hidden, adjacency, dropout) for _ in range(n_layers)]
        elif spatial == "none":
            blocks = [MLPBlock(hidden, dropout) for _ in range(n_layers)]
        else:
            raise ValueError(spatial)
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(nn.Linear(hidden + emb_dim, hidden), nn.ELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_horizons))
        self.register_buffer("node_ids", torch.arange(n_nodes))

    def forward(self, x):                                   # x: (B, L, N, F)
        B, L, N, F = x.shape
        e = self.emb(self.node_ids)                          # (N, E)
        z = torch.cat([x, e.view(1, 1, N, -1).expand(B, L, N, -1)], dim=-1)
        z = Fnn.elu(self.inp(z))                             # (B, L, N, H)
        z = z.permute(0, 2, 1, 3).reshape(B * N, L, -1)
        _, h = self.gru(z)                                   # (1, B*N, H)
        h = h.squeeze(0).view(B, N, -1)
        for blk in self.blocks:
            h = blk(h)
        return self.head(torch.cat([h, e.unsqueeze(0).expand(B, N, -1)], dim=-1))   # (B, N, 3)

    def attention_maps(self):
        """Batch-mean attention of the last evaluated batch, per GAT layer and relation."""
        out = []
        for blk in self.blocks:
            if isinstance(blk, GATBlock):
                out.append([r.last_alpha for r in blk.rel])
        return out


def random_graph_like(adjacency: np.ndarray, seed: int = 0) -> np.ndarray:
    """Same number of out-edges and the same weights per sensor, but to random sensors
    (sanity check: if a random graph works as well, the road graph is not being used)."""
    rng = np.random.default_rng(seed)
    A = adjacency.copy(); np.fill_diagonal(A, 0.0)
    N = A.shape[0]
    R = np.zeros_like(A)
    for i in range(N):
        w = A[i][A[i] > 0]
        if len(w):
            cols = rng.choice(np.setdiff1d(np.arange(N), [i]), size=len(w), replace=False)
            R[i, cols] = w
    np.fill_diagonal(R, 1.0)
    return R
