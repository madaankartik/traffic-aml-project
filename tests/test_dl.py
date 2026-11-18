"""Unit tests for the Phase 2 deep-learning code (no dataset required).

Run with:  python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trafficml.dl.models import EdgeGATLayer, STNet, random_graph_like, relation_masks  # noqa: E402
from trafficml.dl.train import masked_mae_t                                              # noqa: E402


def chain_adj(n=5):
    A = np.zeros((n, n), np.float32)
    for i in range(n - 1):
        A[i, i + 1] = 0.8              # i+1 is downstream of i
    np.fill_diagonal(A, 1.0)
    return A


def test_relation_masks_split_upstream_and_downstream():
    down, up = relation_masks(chain_adj(), directed=True)
    assert down[0, 1] and not down[1, 0], "node 0 receives from its downstream neighbour 1"
    assert up[1, 0] and not up[0, 1], "node 1 receives from its upstream neighbour 0"
    assert down[2, 2] and up[2, 2], "self-loops present"
    (both,) = relation_masks(chain_adj(), directed=False)
    assert both[0, 1] and both[1, 0]


def test_edge_gat_matches_dense_attention_formula():
    torch.manual_seed(0)
    n = 5
    mask = np.eye(n, dtype=bool); mask[0, 1] = mask[0, 2] = mask[3, 4] = True
    w = np.random.default_rng(0).random((n, n)).astype(np.float32)
    layer = EdgeGATLayer(8, 8, 2, mask, w, attn_dropout=0.0).eval()
    h = torch.randn(3, n, 8)
    out = layer(h)
    Wh = layer.W(h).view(3, n, 2, 4)
    s_src, s_dst = (Wh * layer.a_src).sum(-1), (Wh * layer.a_dst).sum(-1)
    e = torch.nn.functional.leaky_relu(s_dst.permute(0, 2, 1).unsqueeze(-1) + s_src.permute(0, 2, 1).unsqueeze(-2), 0.2)
    e = e + layer.beta.view(1, -1, 1, 1) * torch.from_numpy(w) + torch.where(torch.from_numpy(mask), 0.0, float("-inf"))
    ref = torch.einsum("bhij,bjhd->bihd", torch.softmax(e, -1), Wh).reshape(3, n, 8)
    assert torch.allclose(out, ref, atol=1e-5)
    # attention over each node's neighbours sums to one
    a = layer.last_alpha_full[0]                        # (E, heads)
    sums = torch.zeros(n, 2).index_add_(0, layer.dst, a)
    assert torch.allclose(sums, torch.ones(n, 2), atol=1e-5)


def test_no_graph_model_ignores_other_sensors_but_gat_does_not():
    torch.manual_seed(0)
    A = chain_adj()
    x = torch.randn(2, 12, 5, 7)
    x2 = x.clone(); x2[:, :, 4] += 3.0                  # perturb sensor 4 only
    for spatial, should_change in (("none", False), ("gat", True), ("gcn", True)):
        m = STNet(5, A, spatial=spatial, n_layers=2, hidden=16, emb_dim=4, heads=2, dropout=0.0).eval()
        d = (m(x) - m(x2))[:, 2].abs().max().item()    # sensor 2 is two hops from sensor 4
        assert (d > 1e-6) == should_change, spatial


def test_one_layer_gat_sees_exactly_one_hop():
    torch.manual_seed(0)
    A = chain_adj()
    x = torch.randn(1, 12, 5, 7)
    x2 = x.clone(); x2[:, :, 4] += 3.0
    m = STNet(5, A, spatial="gat", n_layers=1, hidden=16, emb_dim=4, heads=2, dropout=0.0).eval()
    diff = (m(x) - m(x2)).abs().amax(dim=(0, 2))
    assert diff[3] > 1e-6 and diff[2] < 1e-7, "one layer = one hop"


def test_masked_loss_ignores_unknown_targets():
    pred = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    y = torch.tensor([[12.0, float("nan")], [float("nan"), 44.0]])
    assert masked_mae_t(pred, y).item() == pytest.approx(3.0)


def test_random_graph_keeps_degree_but_moves_edges():
    A = chain_adj(30)
    R = random_graph_like(A, seed=0)
    np.fill_diagonal(A, 0); Rn = R.copy(); np.fill_diagonal(Rn, 0)
    assert ((A > 0).sum(1) == (Rn > 0).sum(1)).all()
    assert ((A > 0) & (Rn > 0)).sum() < (A > 0).sum()
