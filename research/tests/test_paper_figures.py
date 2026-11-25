"""The automatic check that no figure text touches a line, bar, dot, arrow or another text (analysis/paper_figures.py)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.paper_figures import clashes


def _fig():
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1], [0.5, 0.5], label="flat")
    ax.bar([2], [1.0])
    ax.scatter([3], [0.2], s=10)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 1.2)
    return fig, ax


def test_clear_text_is_not_reported():
    fig, ax = _fig()
    ax.test(0.5, 1.0, "Clear", ha="center", va="center")
    assert clashes(fig) == []
    plt.close(fig)


def test_text_on_a_line_a_bar_a_dot_or_another_text_is_reported():
    fig, ax = _fig()
    ax.text(0.5, 0.5, "On line", ha="center", va="center")
    ax.text(2.0, 0.5, "On bar", ha="center", va="center")
    ax.text(3.0, 0.2, "On dot", ha="center", va="center")
    ax.text(0.5, 1.0, "First", ha="center", va="center")
    ax.text(0.6, 1.0, "Second", ha="center", va="center")
    found = clashes(fig)
    assert any("'On line' touches line flat" in f for f in found)
    assert any("'On bar' touches bar" in f for f in found)
    assert any("'On dot' touches dot" in f for f in found)
    assert any("'First' touches text 'Second'" in f for f in found)
    plt.close(fig)
