import numpy as np
import pytest

pytest.importorskip("matplotlib")

from coralid.explore.clustering import embedding_table  # noqa: E402
from coralid.explore.plots import category_palette, embedding_scatter  # noqa: E402


def test_embedding_scatter_draws_the_table_columns():
    table = embedding_table(
        [f"I.JPG#{i}" for i in range(5)],
        x=[0, 1, 2, 3, 4.0],
        y=[4, 3, 2, 1, 0.0],
        c=[0, 0, 1, 1, 2],
    )
    fig = embedding_scatter(table, color="c", title="map")
    ax = fig.axes[0]
    np.testing.assert_array_equal(ax.collections[0].get_offsets(), table[["x", "y"]].to_numpy())
    np.testing.assert_array_equal(ax.collections[0].get_array(), [0, 0, 1, 1, 2])
    assert ax.get_title() == "map" and ax.get_xlabel() == "x"
    again = embedding_scatter(table, ax=ax)
    assert again is fig  # drawn into the panel it was given


def test_a_category_column_is_drawn_per_category_with_a_legend_that_counts():
    genus = ["Porites", "Acropora", "Porites", "Pavona", "Porites"]
    table = embedding_table(
        [f"I.JPG#{i}" for i in range(5)], x=[0, 1, 2, 3, 4.0], y=[4, 3, 2, 1, 0.0], genus=genus
    )
    fig = embedding_scatter(table, color="genus")
    ax = fig.axes[0]
    assert [c.get_label() for c in ax.collections] == [
        "Acropora (1)",
        "Pavona (1)",
        "Porites (3)",
    ]
    porites = ax.collections[2]
    np.testing.assert_array_equal(porites.get_offsets(), [[0, 4], [2, 2], [4, 0]])
    assert ax.get_legend().get_title().get_text() == "genus"


def test_a_shared_palette_keeps_a_genus_the_same_color_in_every_panel():
    palette = category_palette(["Porites", "Acropora", "Pavona"])
    assert list(palette) == ["Acropora", "Pavona", "Porites"]
    subset = embedding_table(["I.JPG#0", "I.JPG#1"], x=[0, 1.0], y=[0, 1.0], g=["Porites", "Zzz"])
    ax = embedding_scatter(subset, color="g", palette=palette).axes[0]
    drawn = {c.get_label()[:-4]: c.get_facecolor()[0] for c in ax.collections}
    np.testing.assert_allclose(drawn["Porites"][:3], palette["Porites"][:3])
    np.testing.assert_allclose(drawn["Zzz"][:3], [0.827, 0.827, 0.827], atol=0.01)  # not in it


def test_cluster_numbers_sort_as_numbers_and_noise_last():
    palette = category_palette(["10", "2", "noise", "-1", "1"])
    assert list(palette) == ["-1", "1", "2", "10", "noise"]
    many = category_palette([str(i) for i in range(30)])
    assert len(set(many.values())) == 30  # beyond 20 categories the colors are still distinct


def test_a_number_column_gets_a_color_bar_and_a_panel_can_leave_it_out():
    table = embedding_table(
        [f"I.JPG#{i}" for i in range(3)], x=[0, 1, 2.0], y=[2, 1, 0.0], confidence=[0.2, 0.5, 0.9]
    )
    assert len(embedding_scatter(table, color="confidence").axes) == 2  # the map and its bar
    assert len(embedding_scatter(table, color="confidence", legend=False).axes) == 1
    named = table.assign(genus=["A", "B", "B"])
    assert embedding_scatter(named, color="genus", legend=False).axes[0].get_legend() is None
