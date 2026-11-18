"""Draw the Phase 2 architecture diagram (figures/fig_dl_architecture.png)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from trafficml.plots import plt, save, set_style

set_style()
fig, ax = plt.subplots(figsize=(10, 3.9)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.9); ax.axis("off")
def box(x, y, w, h, title, sub, fc="#f0efec", ec="#0b0b0b", tc="#0b0b0b"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=8.6, fontweight="bold", color=tc)
    ax.text(x + w / 2, y + h - 0.52, sub, ha="center", va="top", fontsize=6.8, color=tc, linespacing=1.35)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=1.0, color="#52514e"))
box(0.05, 1.0, 1.55, 1.6, "Input window", "12 steps x 207\nsensors x 7 features:\nspeed (z), mask,\ntime sin/cos,\nusual speed")
box(1.9, 1.0, 1.45, 1.6, "Project+embed", "linear 7+16 -> 64\n+ learned sensor\nembedding (16)")
box(3.65, 1.0, 1.5, 1.6, "Temporal GRU", "one GRU shared by\nall sensors, over the\n12 steps -> h_i (64)")
box(5.45, 1.0, 2.0, 1.6, "Spatial block x K", "none : MLP (no graph)\ngcn  : diffusion, fixed A\ngat  : attention, separate\nupstream / downstream", fc="#0b0b0b", ec="#0b0b0b", tc="white")
box(7.75, 1.0, 1.0, 1.6, "Head", "MLP ->\n3 horizons\n(z units)")
box(9.0, 1.0, 0.95, 1.6, "Forecast", "base +\nsigma_i * z\n(mph)")
for a, b in ((1.6, 1.9), (3.35, 3.65), (5.15, 5.45), (7.45, 7.75), (8.75, 9.0)):
    arrow(a, 1.8, b, 1.8)
box(6.6, 2.7, 3.35, 0.9, "base", "sensor mean (pure DL) or\nXGBoost forecast (hybrid)", fc="#fdecea", ec="#e34948")
arrow(9.47, 2.7, 9.47, 2.6)
box(5.45, 0.05, 2.0, 0.7, "Loss", "masked MAE (+ LWR penalty)", fc="#eef3fb", ec="#2a78d6")
arrow(9.47, 1.0, 7.45, 0.45)
ax.text(5.0, 3.78, "ST-GAT skeleton: only the spatial block and the base change between models", ha="center", fontsize=9, fontweight="bold")
save(fig, "fig_dl_architecture")
