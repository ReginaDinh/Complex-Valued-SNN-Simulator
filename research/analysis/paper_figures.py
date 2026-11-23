"""Figures for the NeurIPS draft (docs/paper/NEURIPS2027_OUTLINE.md, section 4 of the outline), from the raw runs.

    cd research && .venv/bin/python -m analysis.paper_figures [out_dir]

Every panel uses the setting selected in the corresponding report (seeds 0-4 on validation) and all 10 seeds unless noted;
bars are means, error bars 95% CI (t), dots single seeds. Output: fig*.pdf / fig*.png and sources.md in out_dir
(default docs/paper/figures). Every panel is also saved on its own (fig1a_..., fig2a_...), without a panel letter, so it can be
placed separately; the combined figures carry the letters.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COL = {"rot": "#1f4e9c", "norot": "#e08214", "lif_long": "#7f7f7f", "lif_short": "#b8b8b8", "se": "#7b3294", "brf": "#8c510a",
       "adlif": "#c2185b", "gru": "#2ca02c", "lstm": "#98df8a"}
NAME = {"rot": "Rotating cLIF", "norot": "No-rotation cLIF", "lif_long": "LIF (long)", "lif_short": "LIF (short)", "se": "SE-adLIF",
        "brf": "BRF", "adlif": "adLIF (free)", "gru": "GRU (non-spiking)", "lstm": "LSTM (non-spiking)"}
SOURCES = []
BIG = SMALL = 8   # one font size for every text in every figure (figures are drawn at their print size)
FULL, HALF = 5.5, 2.7   # NeurIPS text width and a half-width panel (inches)
plt.rcParams.update({"font.size": BIG, "axes.labelsize": BIG, "xtick.labelsize": SMALL, "ytick.labelsize": SMALL,
                     "legend.fontsize": SMALL, "legend.title_fontsize": SMALL, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "axes.titlepad": 8, "axes.labelpad": 5, "xtick.major.pad": 4, "ytick.major.pad": 4, "legend.borderaxespad": 0.8})


def ci95(x):
    """(mean, half-width of the 95% CI), the same t-based CI as every report (cvsnn.utils.mean_ci95)."""
    from cvsnn.utils import mean_ci95
    m, _, h = mean_ci95([float(v) for v in x])
    return m, h


def load(patterns) -> dict:
    """All runs of the given result directories, deduplicated by key (first occurrence)."""
    rows = {}
    for pat in patterns:
        for f in sorted(glob.glob(str(ROOT / pat), recursive=True)):
            for l in open(f):
                r = json.loads(l)
                rows.setdefault(r["key"], r)
    return rows


def pick(rows: dict, variant: str, lr, seeds=range(10), field=None) -> list:
    lr = lr if isinstance(lr, str) else float(lr)
    got = {r["seed"]: r for r in rows.values() if r["variant"] == variant and (r["lr"] == lr) and r["seed"] in seeds}
    missing = sorted(set(seeds) - set(got))
    if missing:
        raise ValueError(f"{variant} {lr}: missing seeds {missing}")
    rs = [got[s] for s in sorted(got)]
    return [r[field] for r in rs] if field else rs


PANEL = (3.4, 2.6)   # size of a single-panel file (inches)


def bars(ax, groups, ylabel, scale=1.0, legend=True):
    """groups: list of (label, [(model, values)]) -> grouped bars with dots; legend above the axes, clear of the bars."""
    w = 0.8 / max(len(g[1]) for g in groups)
    seen = set()
    for gi, (glabel, items) in enumerate(groups):
        for i, (m, v) in enumerate(items):
            v = np.asarray(v) * scale
            x = gi + (i - (len(items) - 1) / 2) * w
            mu, h = ci95(v)
            ax.bar(x, mu, w * 0.88, color=COL[m], yerr=h, capsize=2, error_kw={"lw": 0.8}, label=None if m in seen else NAME[m])
            seen.add(m)
            ax.scatter(np.full(len(v), x) + np.linspace(-w * 0.22, w * 0.22, len(v)), v, s=4, color="k", alpha=0.6, zorder=3, lw=0)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.03))


CLASHES = []


def clashes(fig, margin=2.5) -> list:
    """Texts that touch a drawn line, bar, marker, arrow, box edge or another text (within `margin` points), measured on the
    rendered figure in display coordinates. Text inside a box is fine as long as it keeps clear of the box edge."""
    from matplotlib.collections import LineCollection, PathCollection, PolyCollection
    from matplotlib.patches import Rectangle
    from matplotlib.text import Annotation, Text
    fig.canvas.draw()
    ren, px = fig.canvas.get_renderer(), fig.dpi / 72

    def dense(xy, step=0.5 * px):
        xy = np.asarray(xy, float)
        if len(xy) < 2:
            return xy
        out = [xy[:1]]
        for a, b in zip(xy[:-1], xy[1:]):
            n = max(1, int(np.hypot(*(b - a)) / step))
            out.append(a + (b - a) * np.linspace(0, 1, n + 1)[1:, None])
        return np.concatenate(out)

    def in_view(ax, t, axis):
        lo, hi = sorted(getattr(ax, f"get_{axis}lim")())
        return lo - 1e-9 <= t.get_position()[0 if axis == "x" else 1] <= hi + 1e-9

    def outline(path):   # a patch outline as dense points, one sub-path at a time (curves flattened, closing vertex skipped)
        polys = [dense(q) for q in path.to_polygons(closed_only=False) if len(q)]
        return np.concatenate(polys) if polys else np.empty((0, 2))

    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    for ax in fig.axes:   # tick labels outside the view limits, or of a hidden axis, are not drawn
        for axis in "xy":
            labels = getattr(ax, f"get_{axis}ticklabels")()
            drawn = [t for t in labels if ax.axison and in_view(ax, t, axis)]
            texts = [t for t in texts if t not in labels or t in drawn]
    boxes = [(t, t.get_window_extent(ren)) for t in texts]
    marks, filled = [], []   # (label, points, radius in px); (label, bbox)
    for ax in fig.axes:
        clip = ax.bbox
        keep = lambda p: p[(p[:, 0] >= clip.x0 - 1) & (p[:, 0] <= clip.x1 + 1) & (p[:, 1] >= clip.y0 - 1) & (p[:, 1] <= clip.y1 + 1)]
        for ln in ax.lines:
            xy = ln.get_transform().transform(np.column_stack(ln.get_data()).astype(float))
            xy = xy[np.isfinite(xy).all(1)]
            if ln.get_linestyle() not in ("None", "", " "):
                marks.append((f"line {ln.get_label()}", keep(dense(xy)), 0.5 * ln.get_linewidth() * px))
            if ln.get_marker() not in ("None", "", " ", None):
                marks.append((f"marker {ln.get_label()}", keep(xy), 0.5 * ln.get_markersize() * px))
        for c in ax.collections:
            if isinstance(c, LineCollection):
                pts = [dense(c.get_transform().transform(s)) for s in c.get_segments() if len(s)]
                if pts:
                    marks.append(("error bar", keep(np.concatenate(pts)), 0.5 * px))
            elif isinstance(c, PathCollection):
                xy = c.get_offset_transform().transform(c.get_offsets())
                marks.append(("dot", keep(xy), 0.5 * np.sqrt(np.max(c.get_sizes()) if len(c.get_sizes()) else 1) * px))
            elif isinstance(c, PolyCollection):
                for p in c.get_paths():
                    marks.append(("band", keep(dense(c.get_transform().transform(p.vertices))), 0))
        for p in ax.patches:
            if isinstance(p, Rectangle) and p.get_fill():
                filled.append(("bar", p.get_window_extent(ren)))
            else:
                marks.append((type(p).__name__, outline(p.get_transform().transform_path(p.get_path())), 0.5 * (p.get_linewidth() or 0) * px))
        for t in ax.texts:
            if isinstance(t, Annotation) and t.arrow_patch is not None:
                a = t.arrow_patch
                marks.append(("arrow", outline(a.get_transform().transform_path(a.get_path())), 0.5 * a.get_linewidth() * px))
        for s in ax.spines.values():
            if ax.axison and s.get_visible():
                marks.append(("axis line", outline(s.get_transform().transform_path(s.get_path())), 0.5 * s.get_linewidth() * px))
    found, m = [], margin * px
    for i, (t, b) in enumerate(boxes):
        x0, y0, x1, y1 = b.x0 - m, b.y0 - m, b.x1 + m, b.y1 + m
        for label, pts, rad in marks:
            if len(pts) and ((pts[:, 0] >= x0 - rad) & (pts[:, 0] <= x1 + rad) & (pts[:, 1] >= y0 - rad) & (pts[:, 1] <= y1 + rad)).any():
                found.append(f"'{t.get_text()}' touches {label}")
        for label, fb in filled:
            if fb.x0 < x1 and fb.x1 > x0 and fb.y0 < y1 and fb.y1 > y0:
                found.append(f"'{t.get_text()}' touches {label}")
        for u, c in boxes[i + 1:]:
            if c.x0 < x1 and c.x1 > x0 and c.y0 < y1 and c.y1 > y0:
                found.append(f"'{t.get_text()}' touches text '{u.get_text()}'")
    return sorted(set(found))


def save(fig, out: Path, name: str, src: str | None = None):
    for c in clashes(fig):
        CLASHES.append(f"{name}: {c}")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    if src:
        SOURCES.append(f"- `{name}`: {src}")


def emit(out: Path, name: str, src: str, panels: list, layout, shared=(), legend_title=None, ncol=3):
    """panels: [(file name, function(fig, spec, own_legend), single-panel size, caption title)]; layout() -> (fig, specs).
    Single-panel files: own legend, no letter, no title (the paper's subcaption adds them). Combined figure: one legend
    above the whole figure when every panel shows the same entries (`shared` = all panels), otherwise each panel keeps
    its own; "(a) title" is centred on each panel's x axis, on one baseline per row."""
    from matplotlib.transforms import Bbox
    for fname, fn, size, _ in panels:
        fig = plt.figure(figsize=size, layout="constrained")
        fn(fig, fig.add_gridspec(1, 1)[0], True)
        save(fig, out, fname)
    fig, specs = layout()
    groups = []
    for k, ((fname, fn, _, title), spec, letter) in enumerate(zip(panels, specs, "abcdefgh")):
        n0 = len(fig.axes)
        fn(fig, spec, k not in shared)
        groups.append((fig.axes[n0:], f"({letter}) {title}"))
    fig.canvas.draw()
    fig.set_layout_engine("none")   # freeze the layout before adding captions and the shared legend
    renderer, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    tight = [Bbox.union([a.get_tightbbox(renderer) for a in axs]).transformed(inv) for axs, _ in groups]
    plot = [Bbox.union([a.get_position() for a in axs]) for axs, _ in groups]
    pt = 1 / (fig.get_figheight() * 72)
    for k, (axs, cap) in enumerate(groups):
        row = [b for b in tight if b.y0 < tight[k].y1 and b.y1 > tight[k].y0]
        fig.text((plot[k].x0 + plot[k].x1) / 2, min(b.y0 for b in row) - 10 * pt, cap, ha="center", va="top", fontsize=BIG)
    if shared:
        handles, labels = [], []
        for k in shared:
            for a in groups[k][0]:
                for h, l in zip(*a.get_legend_handles_labels()):
                    if l not in labels:
                        handles.append(h)
                        labels.append(l)
        xs = Bbox.union([plot[k] for k in shared])
        top = max(b.y1 for b in tight)
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=((xs.x0 + xs.x1) / 2, top + 4 * pt), bbox_transform=fig.transFigure,
                   ncol=ncol, frameon=False, title=legend_title)
    save(fig, out, name, src + "; single panels: " + ", ".join(f"`{p[0]}`" for p in panels))


def _two_rows(height):
    """Two half-width panels on top, one full-width panel below."""
    def layout():
        fig = plt.figure(figsize=(FULL, height), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.05, h_pad=0.05, hspace=0.02, wspace=0.08)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.2, 1])   # an empty middle row separates the two rows
        return fig, [gs[0, 0], gs[0, 1], gs[2, :]]
    return layout


def _row(n, size, widths=None):
    def layout():
        fig = plt.figure(figsize=size, layout="constrained")
        fig.get_layout_engine().set(w_pad=0.08, h_pad=0.08, wspace=0.08)
        return fig, list(fig.add_gridspec(1, n, width_ratios=widths))
    return layout


# ---------- figure 1: model and intuition
def p1a_schematic(fig, spec, own):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    ax = fig.add_subplot(spec)
    ax.set_xlim(0, 18.6)
    ax.set_ylim(0.75, 8.35)
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", fc=fc, ec="k", lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=BIG, linespacing=1.5)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.8, color="k"))

    box(0.15, 3.3, 2.4, 1.6, "Input\nspikes $x_t$", "#f2f2f2")
    box(4.1, 2.4, 5.0, 3.4, "Polar cLIF neurons\n" + r"$U_t = r\,e^{i\omega} U_{t-1} + W x_t$" + "\n" + r"Spike if Re $U_t > \theta$" + "\n"
        + r"$r = \sigma(\rho)$, $\omega$ learned", "#dbe5f5")
    box(11.0, 5.6, 5.2, 2.4, "Plastic synapse $A_t$\nGated delta rule:\nwrite $(k_t, v_t)$, read $q_t$", "#fbe3c9")
    box(14.9, 1.9, 3.3, 2.3, "Read-out\n(leaky units)", "#f2f2f2")
    arrow(2.8, 4.1, 3.9, 4.1)
    arrow(9.35, 5.3, 10.8, 6.5)
    ax.annotate("$k_t, v_t, q_t$", (10.05, 5.9), xytext=(-10, 6), textcoords="offset points", ha="right", va="bottom", fontsize=SMALL)
    arrow(9.35, 3.0, 14.6, 3.0)
    ax.annotate("Spikes $h_t$ (0/1)", (11.95, 3.0), xytext=(0, -8), textcoords="offset points", ha="center", va="top", fontsize=SMALL)
    arrow(15.6, 5.4, 15.6, 4.45)
    ax.annotate("$y_t = A_t q_t$", (15.6, 4.92), xytext=(8, 0), textcoords="offset points", ha="left", va="center", fontsize=SMALL)
    ax.text(9.3, 1.15, "Complex numbers only inside the neuron; synapses, spikes and the plastic synapse are real.", fontsize=SMALL,
            ha="center", va="center")


def p1b_phasor(fig, spec, own):
    sub = spec.subgridspec(2, 1, height_ratios=[0.3, 2.0], hspace=0.12)
    for k, (om, title, col, lims) in enumerate(((0.0, "No rotation", COL["norot"], ((-0.5, 18.5), (-0.5, 0.5))),
                                                (0.5, "Rotation, $\\omega = 0.5$", COL["rot"], ((-1.3, 3.1), (-1.0, 3.6))))):
        a2 = fig.add_subplot(sub[k])
        z = 0j
        for t in range(40):
            step = 0.95 ** t * np.exp(1j * om * t)
            a2.annotate("", xy=(z.real + step.real, z.imag + step.imag), xytext=(z.real, z.imag),
                        arrowprops=dict(arrowstyle="-|>", lw=0.7, color=col, mutation_scale=5, shrinkA=0, shrinkB=0))
            if om == 0:
                a2.plot([z.real], [z.imag], "|", color=col, ms=4, mew=0.6)
            z += step
        a2.annotate("", xy=(z.real, z.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=1.1, color="k", ls="--", mutation_scale=6))
        a2.set_xlim(*lims[0])
        a2.set_ylim(*lims[1])
        if om:
            a2.set_aspect("equal")
        a2.axis("off")
        a2.set_title(f"{title}: |sum| = {abs(z):.1f}", fontsize=SMALL, loc="center", pad=6)


def p1c_neuron(fig, spec, own):
    a3 = fig.add_subplot(spec)
    rng = np.random.default_rng(0)
    I = 0.03 + 0.03 * rng.standard_normal(400)
    for om, col, lab in ((0.0, COL["norot"], r"$\omega = 0$"), (0.15, COL["rot"], r"$\omega = 0.15$")):
        u, tr = 0j, []
        for t in range(400):
            u = 0.99 * np.exp(1j * om) * u + I[t]
            tr.append(u.real)
        a3.plot(tr, color=col, lw=0.9, label=lab)
    a3.axhline(0.5, color="k", lw=0.6, ls=":")
    a3.annotate("Threshold", (1.0, 0.5), xycoords=a3.get_yaxis_transform(), xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=SMALL)
    a3.set_xlabel("Time step")
    a3.set_ylabel("Re $U$ (no reset)")
    a3.set_ylim(-0.7, 3.4)
    a3.legend(frameon=False, loc="center right")


def fig1(out: Path):
    def layout():
        fig = plt.figure(figsize=(FULL, 5.3), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.05, h_pad=0.05, hspace=0.02, wspace=0.08)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 0.2, 1])   # an empty middle row separates (a) from (b), (c)
        return fig, [gs[0, :], gs[2, 0], gs[2, 1]]
    emit(out, "fig1_model", "schematic (a); illustration (b): head-to-tail sum of r^t e^{iwt} over 40 steps, r = 0.95, w = 0 vs 0.5; "
         "(c) one neuron without reset, r = 0.99, input mean 0.03 + noise, w = 0 vs 0.15 (Proposition 1)",
         [("fig1a_schematic", p1a_schematic, (FULL, 2.5), "Model"), ("fig1b_phasor_sum", p1b_phasor, (HALF, 2.6), "Sum of the last 40 inputs, $r = 0.95$"),
          ("fig1c_one_neuron", p1c_neuron, (HALF, 2.3), "One neuron without reset, $r = 0.99$")], layout)


# ---------- figure 2: rotation x memory (synthetic)
def _fig2_data():
    rows = load(["results/ei54*/**/runs.jsonl", "results/ei61c/**/runs.jsonl"])
    cells = {"rot_short": ("cand_single_slowfw|2heads|phaselr", "0.01/0.0003"), "norot_short": ("cand_norot_slowfw|2heads|phaselr", "0.01/0.0003"),
             "rot_long": ("cand_long_fw003|2heads|phaselr", "0.01/0.0003"), "norot_long": ("cand_longnorot_fw001|2heads|phaselr", "0.1/0.0003"),
             "lif_short": ("lif_short_fw003|2heads|phaselr", "0.03/0.0003"), "lif_long": ("lif_fw2_d11_fw001|2heads|phaselr", "0.1/0.001")}
    return {k: pick(rows, *v) for k, v in cells.items()}


def _fig2_groups(d, field):
    return [("Short memory\n(reset)", [("rot", [r[field] for r in d["rot_short"]]), ("norot", [r[field] for r in d["norot_short"]]),
                                        ("lif_short", [r[field] for r in d["lif_short"]])]),
            ("Long memory\n(no reset)", [("rot", [r[field] for r in d["rot_long"]]), ("norot", [r[field] for r in d["norot_long"]]),
                                         ("lif_long", [r[field] for r in d["lif_long"]])])]


def fig2(out: Path):
    d = _fig2_data()

    def pa(fig, spec, own):
        ax = fig.add_subplot(spec)
        bars(ax, _fig2_groups(d, "ecl_forgetting"), "Later-task forgetting (%)", 100, legend=own)
        ax.set_ylim(0, 25)

    def pb(fig, spec, own):
        ax = fig.add_subplot(spec)
        bars(ax, _fig2_groups(d, "recall_final"), "Recall at the end", 1, legend=own)
        ax.set_ylim(0.0, 1.08)
        ax.axhline(0.125, color="k", lw=0.5, ls=":")
        ax.annotate("Chance", (1.0, 0.125), xycoords=ax.get_yaxis_transform(), xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=SMALL)
    emit(out, "fig2_rotation_x_memory", "synthetic sequence (recall 8 pairs, then 5 E-CL tasks), two read-outs; I54i (cLIF 2 x 2), I54g (LIF long), "
         "I54j (LIF short); 10 seeds; settings as selected in those reports",
         [("fig2a_forgetting", pa, (HALF, 2.5), "Forgetting of the later tasks"), ("fig2b_recall", pb, (HALF, 2.5), "Recall at the end")], _row(2, (FULL, 2.6)),
         shared=(0, 1), ncol=2)


# ---------- figure 3: parameterisation 2 x 2
def fig3(out: Path):
    rows = load(["results/ei61c/**/runs.jsonl"])
    cells = {("polar", "Frozen"): ("cand_long_frz_fw003|2heads|phaselr", "0.03/0.001"), ("polar", "Learned"): ("cand_long_fw003|2heads|phaselr", "0.01/0.0003"),
             ("free", "Frozen"): ("adlif_long_frz_slowfw|2heads|phaselr", "0.03/0.003"), ("free", "Learned"): ("adlif_long_slowfw|2heads|phaselr", "0.01/0.0003")}
    d = {k: pick(rows, *v) for k, v in cells.items()}

    def panel(field, lab, title):
        def fn(fig, spec, own):
            ax = fig.add_subplot(spec)
            w = 0.36
            for i, form in enumerate(("polar", "free")):
                for j, state in enumerate(("Frozen", "Learned")):
                    v = np.asarray([r[field] for r in d[(form, state)]])
                    x = i + (j - 0.5) * w
                    mu, h = ci95(v)
                    ax.bar(x, mu, w * 0.88, color=COL["rot" if i == 0 else "adlif"], alpha=0.45 if state == "Frozen" else 1.0, yerr=h, capsize=2,
                           error_kw={"lw": 0.8}, label=state if i == 0 else None)
                    ax.scatter(np.full(len(v), x) + np.linspace(-0.08, 0.08, len(v)), v, s=4, color="k", alpha=0.6, zorder=3, lw=0)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Polar form\n(cLIF)", "Free form\n(adLIF)"])
            ax.set_ylabel(lab)
            ax.set_ylim(0, 1.08)
            if field == "recall_final":
                ax.axhline(0.125, color="k", lw=0.5, ls=":")
                ax.annotate("Chance", (1.0, 0.125), xycoords=ax.get_yaxis_transform(), xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=SMALL)
            if own:
                ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02), title="Neuron parameters")
        return fn
    emit(out, "fig3_parameterization", "I61c + N11 (frozen cells after the grid-edge extension): identical initial dynamics, neuron parameters frozen or learned; 10 seeds",
         [("fig3a_recall", panel("recall_final", "Recall at the end", "Recall"), (HALF, 2.5), "Recall"),
          ("fig3b_validation_score", panel("val_score", "Validation score", "Validation score"), (HALF, 2.5), "Validation score")], _row(2, (FULL, 2.7)),
         shared=(0, 1), legend_title="Neuron parameters", ncol=2)


# ---------- figure 4: ten tasks
def fig4(out: Path):
    rows = load(["results/ei69/**/runs.jsonl"])
    cells = {"rot": ("cand_long_fw003", "0.01/0.001"), "norot": ("cand_longnorot_fw001", "0.1/0.003"), "lif_long": ("lif_fw2_d11_fw001", "0.1/0.003"),
             "lif_short": ("lif_short_fw003", "0.03/0.003"), "se": ("se_adlif_fw001", "0.1/0.001"), "brf": ("brfv2_fw003", "0.01/0.003")}
    fig, ax = plt.subplots(figsize=(FULL, 2.5), layout="constrained")
    t = np.arange(1, 11)
    for m, (v, lr) in cells.items():
        a = np.asarray(pick(rows, f"{v}|2heads|phaselr|mnist|h120|rp100|t10", lr, field="seen_avg"))
        mu = a.mean(0)
        h = np.asarray([ci95(a[:, k])[1] for k in range(a.shape[1])])
        ax.plot(t, mu, color=COL[m], label=NAME[m], lw=1.6 if m == "rot" else 1.0, zorder=3 if m == "rot" else 2)
        ax.fill_between(t, mu - h, mu + h, color=COL[m], alpha=0.07 if m == "brf" else 0.18, lw=0)
    ax.set_xlabel("Tasks learned")
    ax.set_ylabel("Mean accuracy, tasks so far")
    ax.set_xticks(t)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.03, 0.5))
    save(fig, out, "fig4_pmnist_10tasks", "I69: permuted MNIST, 10 tasks, experience replay (100 samples per task), 120 hidden neurons; each model at its "
         "I64b / I67 selected later-phase lr; 10 seeds, band = 95% CI")


# ---------- figure 5: classification vs continual learning
def fig5(out: Path):
    shd = load(["results/ei68/brfv2_cmp/runs.jsonl", "results/ei66/long80_all/runs.jsonl", "results/ei66/long80/*/runs.jsonl"])   # the last: N21 seeds 5-9
    cl = load(["results/ei67/shd_vs_i64a/runs.jsonl", "results/ei67/mnist_vs_i64b/runs.jsonl"])
    std = {"rot": ("cand_long_fw003", 0.001, 10), "se": ("se_adlif_fw001", 0.001, 10), "brf": ("brfv2_fw003", 0.001, 10),
           "lif_short": ("lif_short_fw003", 0.003, 10), "norot": ("cand_longnorot_fw001", 0.01, 10), "lif_long": ("lif_fw2_d11_fw001", 0.001, 10)}   # N21: seeds 5-9 added
    sel = {"shd": {"rot": "0.01/0.001", "norot": "0.1/0.001", "lif_long": "0.1/0.001", "lif_short": "0.03/0.001", "se": "0.1/0.003", "brf": "0.01/0.001"},
           "mnist": {"rot": "0.01/0.001", "norot": "0.1/0.003", "lif_long": "0.1/0.003", "lif_short": "0.03/0.003", "se": "0.1/0.001", "brf": "0.01/0.003"}}
    var = {"rot": "cand_long_fw003", "norot": "cand_longnorot_fw001", "lif_long": "lif_fw2_d11_fw001", "lif_short": "lif_short_fw003", "se": "se_adlif_fw001", "brf": "brfv2_fw003"}

    def panel(task, title, legend):
        def fn(fig, spec, own):
            ax = fig.add_subplot(spec)
            for m, (v, lr, n) in std.items():
                xs = pick(shd, v, lr, range(n), field="test")
                ys = pick(cl, f"{var[m]}|2heads|phaselr|{task}|h120|rp100", sel[task][m], field="ecl_final_avg")
                (mx, hx), (my, hy) = ci95(xs), ci95(ys)
                ax.errorbar(mx, my, xerr=hx, yerr=hy, fmt="o", color=COL[m], ms=4, capsize=2, lw=0.8, label=NAME[m] + ("" if n == 10 else " (SHD: 5 seeds)"))
            ax.set_xlim(0.755, 0.895)
            ax.set_xlabel("Standard SHD test accuracy\n(one-shot classification)")
            ax.set_ylabel("Final accuracy\n(continual learning)")
            if legend and own:   # single-panel file: legend below the plot, managed by the layout
                h, l = ax.get_legend_handles_labels()
                fig.legend(h, l, loc="outside lower center", ncol=2, frameon=False)
        return fn
    emit(out, "fig5_classification_vs_continual", "x: standard SHD, 80 epochs, I66 / I68 / N21, 10 seeds per model; "
         "y: I64a / I67 (Split-SHD) and I64b / I67 (permuted MNIST) with replay, selected settings, 10 seeds; error bars 95% CI",
         [("fig5a_split_shd", panel("shd", "Split-SHD + replay", True), (HALF, 3.2), "Split-SHD with replay"),
          ("fig5b_permuted_mnist", panel("mnist", "Permuted MNIST + replay", True), (HALF, 3.2), "Permuted MNIST with replay")],
         _row(2, (FULL, 2.8)), shared=(0, 1), ncol=2)


# ---------- figure 6: mechanism
def fig6(out: Path):
    import torch

    def pa(fig, spec, own):
        ax = fig.add_subplot(spec)
        r = np.linspace(0.5, 0.999, 400)
        for om, ls in ((0.0, "-"), (0.15, "--"), (0.5, ":")):
            ax.plot(r, 1 / np.abs(1 - r * np.exp(1j * om)), ls, color=COL["rot"] if om else COL["norot"], label=rf"$\omega = {om:g}$")
        ax.axhline(1 / np.sin(0.15), color=COL["rot"], lw=0.5, alpha=0.6, label="Bound $1/\\sin\\omega$\n" r"($\omega = 0.15$)")
        ax.set_yscale("log")
        ax.set_xlabel(r"Memory $r = |\lambda|$")
        ax.set_ylabel(r"Pile-up gain $1/|1-\lambda|$")
        leg = ax.legend(frameon=False, loc="upper left")
        for column in leg._legend_handle_box.get_children():   # centre each line sample on its label (the bound's label has two lines)
            for entry in column.get_children():
                entry.align = "center"

    def pb(fig, spec, own):
        ax = fig.add_subplot(spec)
        bins = np.linspace(0, 1, 26)
        for cell, d, key in (("rot", "results/n10", "rotating long"), ("norot", "results/n12", "no-rotation long"), ("lif_long", "results/n12", "LIF long")):
            rs = []
            for l in (ROOT / d / "runs.jsonl").read_text().splitlines():
                row = {"cell": "rotating long", **json.loads(l)}
                if row["cell"] == key and row["setting"] == "synthetic (I54i)":
                    rs.append(torch.load(row["save_model"], weights_only=False)["final"][0]["r"].numpy())
            ax.hist(np.concatenate(rs), bins=bins, histtype="step", color=COL[cell], label=NAME[cell], lw=1.2)
        ax.set_xlabel("Memory $r$ after training")
        ax.set_ylabel("Neurons (10 seeds)")
        ax.set_ylim(0, 80 if own else 52)
        if own:
            ax.legend(frameon=False, loc="upper right")

    def pc(fig, spec, own):
        ax = fig.add_subplot(spec)
        syn = json.loads((ROOT / "results/n17/activity.json").read_text())
        ben = json.loads((ROOT / "results/n18/activity.json").read_text())
        ks = (("rot", "rotating long"), ("norot", "no-rotation long"), ("lif_long", "LIF long"))
        groups = [("Synthetic", {c: syn[k]["long"]["mag"] for c, k in ks}),
                  ("MNIST\nrecall", {c: ben[f"permuted MNIST + replay (I64b)|recall|{k}"]["long_mag"] for c, k in ks}),
                  ("SHD\nrecall", {c: ben[f"Split-SHD + replay (I64a)|recall|{k}"]["long_mag"] for c, k in ks}),
                  ("SHD\ntask data", {c: ben[f"Split-SHD + replay (I64a)|task 4|{k}"]["long_mag"] for c, k in ks})]
        w = 0.26
        for gi, (g, vals) in enumerate(groups):
            for i, c in enumerate(("rot", "norot", "lif_long")):
                ax.bar(gi + (i - 1) * w, vals[c], w * 0.88, color=COL[c], label=NAME[c] if gi == 0 else None)
        ax.axhline(1.0, color="k", lw=0.6, ls=":")
        ax.annotate("Threshold", (1.0, 1.0), xycoords=ax.get_yaxis_transform(), xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=SMALL)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g for g, _ in groups])
        ax.set_ylabel("Mean $|U|$ / threshold")
        ax.set_ylim(0, 31 if own else 23)
        if own:
            ax.legend(frameon=False, loc="upper right")
    emit(out, "fig6_mechanism", "(a) Proposition 1, exact formula; (b) N10 / N12, synthetic sequence, selected settings, 10 seeds pooled; "
         "(c) N17 (synthetic) and N18 (benchmark-trained models, no new training), selected settings, 10 seeds pooled; dotted line = threshold",
         [("fig6a_proposition1", pa, (HALF, 2.4), "Proposition 1"), ("fig6b_memory_after_training", pb, (HALF, 2.4), "Memory after training (synthetic)"),
          ("fig6c_long_memory_potential", pc, (FULL, 2.3), r"Long-memory neurons ($r \geq 0.8$)")], _two_rows(4.9))



def table1(out: Path):
    """Two benchmarks x {no method, SI, replay}: final accuracy (mean ± 95% CI, 10 seeds) and seeds where the rotating
    network is higher; each cell at the setting selected in its report (I59, I59b, I63, I63b, I64a, I64b, I67)."""
    rows = load(["results/ei59*/**/runs.jsonl", "results/ei63*/**/runs.jsonl", "results/ei64/**/runs.jsonl", "results/ei67/**/runs.jsonl",
                 "results/n19/mnist/**/runs.jsonl", "results/n19/shd/**/runs.jsonl"])
    V = {"rot": "cand_long_fw003", "norot": "cand_longnorot_fw001", "lif_long": "lif_fw2_d11_fw001", "lif_short": "lif_short_fw003",
         "se": "se_adlif_fw001", "brf": "brfv2_fw003", "gru": "gru_fw001", "lstm": "lstm_fw003"}
    LR = {"rot": "0.01", "norot": "0.1", "lif_long": "0.1", "lif_short": "0.03", "se": "0.1", "brf": "0.01", "gru": "0.01", "lstm": "0.01"}
    SPIKING = ("rot", "norot", "lif_long", "lif_short", "se", "brf")
    sel = {  # (task, method): {model: (variant suffix, later-phase lr)}
        ("mnist", "none"): {"rot": ("mnist|h120", "0.0001"), "norot": ("mnist|h120", "0.0003"), "lif_long": ("mnist|h120", "0.0003"), "lif_short": ("mnist|h120", "0.001")},
        ("mnist", "SI"): {"rot": ("mnist|si=10|h120", "0.001"), "norot": ("mnist|si=10|h120", "0.001"), "lif_long": ("mnist|si=10|h120", "0.001"), "lif_short": ("mnist|si=30|h120", "0.001")},
        ("mnist", "replay"): {"rot": ("mnist|h120|rp100", "0.001"), "norot": ("mnist|h120|rp100", "0.003"), "lif_long": ("mnist|h120|rp100", "0.003"),
                              "lif_short": ("mnist|h120|rp100", "0.003"), "se": ("mnist|h120|rp100", "0.001"), "brf": ("mnist|h120|rp100", "0.003"),
                              "gru": ("mnist|h120|rp100", "0.001"), "lstm": ("mnist|h120|rp100", "0.001")},
        ("shd", "none"): {"rot": ("shd|h120", "3e-05"), "norot": ("shd|h120", "0.0003"), "lif_long": ("shd|h120", "0.003"), "lif_short": ("shd|h120", "0.0003")},
        ("shd", "SI"): {"rot": ("shd|si=100|h120", "0.001"), "norot": ("shd|si=100|h120", "0.001"), "lif_long": ("shd|si=30|h120", "0.001"), "lif_short": ("shd|si=10|h120", "0.001")},
        ("shd", "replay"): {"rot": ("shd|h120|rp100", "0.001"), "norot": ("shd|h120|rp100", "0.001"), "lif_long": ("shd|h120|rp100", "0.001"),
                            "lif_short": ("shd|h120|rp100", "0.001"), "se": ("shd|h120|rp100", "0.003"), "brf": ("shd|h120|rp100", "0.001"),
                            "gru": ("shd|h120|rp100", "0.001"), "lstm": ("shd|h120|rp100", "0.0003")}}
    cols = [("mnist", "none"), ("mnist", "SI"), ("mnist", "replay"), ("shd", "none"), ("shd", "SI"), ("shd", "replay")]
    vals = {}
    for col, cells in sel.items():
        for m, (suf, lre) in cells.items():
            vals[(col, m)] = pick(rows, f"{V[m]}|2heads|phaselr|{suf}", f"{LR[m]}/{lre}", field="ecl_final_avg")
    md = ["| | PMNIST none | PMNIST SI | PMNIST replay | Split-SHD none | Split-SHD SI | Split-SHD replay |", "|---|---|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{lcccccc}", r"\toprule", r" & \multicolumn{3}{c}{Permuted MNIST} & \multicolumn{3}{c}{Split-SHD} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}", r"Neuron & none & SI & replay & none & SI & replay \\", r"\midrule"]
    for m in ("rot", "norot", "lif_long", "lif_short", "se", "brf", "gru", "lstm"):
        if m == "gru":
            md.append("| _non-spiking reference_ | | | | | | |")
            tex.append(r"\midrule")
        cm, ct = [], []
        for col in cols:
            if (col, m) not in vals:
                cm.append("–")
                ct.append("--")
                continue
            mu, h = ci95(vals[(col, m)])
            best = m in SPIKING and max(ci95(vals[(col, k)])[0] for k in sel[col] if k in SPIKING) == mu
            wins = "" if m == "rot" else f" ({sum(a > b for a, b in zip(vals[(col, 'rot')], vals[(col, m)]))}/10)"
            cm.append((f"**{mu:.3f}**" if best else f"{mu:.3f}") + f" ± {h:.3f}{wins}")
            ct.append((rf"\textbf{{{mu:.3f}}}" if best else f"{mu:.3f}") + rf"{{\scriptsize$\pm${h:.3f}}}" + (f"$^{{{wins.strip(' ()')}}}$" if wins else ""))
        md.append(f"| {NAME[m]} | " + " | ".join(cm) + " |")
        tex.append(f"{NAME[m]} & " + " & ".join(ct) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "table1.md").write_text("# Table 1 (generated by research/analysis/paper_figures.py)\n\n"
                                    "Final accuracy after the continual-learning sequence (mean ± 95% CI, 10 seeds; bold = best spiking network in column). "
                                    "In brackets: seeds (of 10) in which the rotating cLIF is higher. Each cell at the setting selected in its report "
                                    "(I59, I59b, I63, I63b, I64a, I64b, I67, N19); SE-adLIF, BRF and the non-spiking GRU / LSTM (same plastic synapse, "
                                    "parameter-matched) were run with replay only.\n\n" + "\n".join(md) + "\n")
    (out / "table1.tex").write_text("% generated by research/analysis/paper_figures.py; needs \\usepackage{booktabs}\n" + "\n".join(tex) + "\n")
    SOURCES.append("- `table1`: I59, I59b, I63, I63b, I64a, I64b, I67, N19 (GRU / LSTM); final accuracy (ecl_final_avg), selected settings, 10 seeds")


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, table1):
        f(out)
    (out / "sources.md").write_text("# Figure sources (generated by research/analysis/paper_figures.py)\n\n" + "\n".join(SOURCES) +
                                    "\n")
    print("\n".join(SOURCES))
    print(f"\nText clashes: {len(CLASHES)}", *CLASHES, sep="\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "docs/paper/figures")
