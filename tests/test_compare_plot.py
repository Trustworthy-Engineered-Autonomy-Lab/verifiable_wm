import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

import compare


def test_draw_initial_cell_adds_green_outline_for_each_wrapped_interval_pair():
    original_dims = compare.PLOT_DIMS
    compare.PLOT_DIMS = (0, 1)
    fig, ax = plt.subplots()
    try:
        compare.draw_initial_cell(
            ax,
            [[-1.0, 1.0], [-3.14, -3.0, 3.0, 3.14]],
        )

        assert len(ax.patches) == 2
        assert all(patch.get_edgecolor() == to_rgba("green") for patch in ax.patches)
        assert all(not patch.get_fill() for patch in ax.patches)
    finally:
        compare.PLOT_DIMS = original_dims
        plt.close(fig)
