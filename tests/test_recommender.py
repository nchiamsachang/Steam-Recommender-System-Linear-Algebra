"""Tests for the recommender, run on the synthetic dataset described in conftest.py.

Cosine similarity between two games is (users who recommended both) divided by
sqrt(recommenders of one * recommenders of the other). For Stardew Valley (4 recommenders):

    Terraria                          3 shared / sqrt(4 * 3) = 0.87
    Starbound                         2 shared / sqrt(4 * 2) = 0.71
    The Witcher 3: Wild Hunt (GOTY)   1 shared / sqrt(4 * 2) = 0.35
    Hades                             1 shared / sqrt(4 * 3) = 0.29
    Portal, Portal 2, Half-Life 2     0 shared               = 0
"""
import pytest

from conftest import recommender

STARDEW_NEIGHBOURS = {"Terraria", "Starbound", "The Witcher 3: Wild Hunt (GOTY)", "Hades"}


def test_matrix_has_one_row_per_reviewed_game_and_one_column_per_reviewer(ratings):
    X, game_mapper, game_inv_mapper = recommender.create_matrix(ratings)

    # 8 games have both metadata and reviews; the two that lack one of them are left out.
    # 9 users reviewed those games. Users 2 and 9 count although they gave a thumbs-down;
    # user 10 only reviewed the game without metadata and does not.
    assert X.shape == (8, 9)
    assert len(game_mapper) == 8
    assert all(game_inv_mapper[row] == app_id for app_id, row in game_mapper.items())
    # a cell is 1 only where the user recommended the game: 25 of the 27 reviews
    assert X.astype(int).sum() == 25
    stardew = X[game_mapper[103]].toarray().ravel()
    assert stardew.sum() == 4


def test_known_game_gets_five_recommendations(catalog):
    recommendations = catalog.recommend("Stardew Valley")

    assert len(recommendations) == 5
    # The four games that share recommenders with it must all be there. The fifth place is
    # a three-way tie at similarity 0, so which of those games fills it is not asserted.
    assert STARDEW_NEIGHBOURS <= set(recommendations)
    assert set(recommendations) - STARDEW_NEIGHBOURS <= {"Portal", "Portal 2", "Half-Life 2"}


def test_recommendations_leave_out_the_queried_game_and_repeat_nothing(catalog):
    for title in sorted(set(catalog.ratings["title"])):
        recommendations = catalog.recommend(title)
        assert title not in recommendations, f"{title} was recommended to someone who entered {title}"
        assert len(recommendations) == len(set(recommendations))
        assert len(recommendations) == 5


def test_a_game_with_identical_reviewers_is_recommended_not_mistaken_for_the_query(catalog):
    # Portal and Portal 2 were recommended by exactly the same users, so each is at
    # distance 0 from the other as well as from itself.
    assert "Portal 2" in catalog.recommend("Portal")
    assert "Portal" in catalog.recommend("Portal 2")


def test_missing_title_gives_no_recommendations(catalog, capsys):
    recommendations = catalog.recommend("A Game That Is Not In The Dataset")

    assert recommendations == []
    assert "not found" in capsys.readouterr().out


@pytest.mark.parametrize(
    "search, expected",
    [
        ("portal", ["Portal", "Portal 2"]),         # part of a title, any case
        ("STAR", ["Stardew Valley", "Starbound"]),
        ("witcher", ["The Witcher 3: Wild Hunt (GOTY)"]),
        ("3: wild", ["The Witcher 3: Wild Hunt (GOTY)"]),
        ("zzz", []),
        ("Game Without Metadata", []),              # dropped when the data was loaded
    ],
)
def test_partial_title_search_finds_the_expected_games(ratings, search, expected):
    assert list(recommender.find_title_matches(ratings, search)) == expected


@pytest.mark.parametrize("search", ["(GOTY", "(", "3: Wild Hunt (", "[", "+", "\\"])
def test_search_text_is_matched_literally(ratings, search):
    # Titles contain brackets and other punctuation; typing them must search, not crash.
    matches = list(recommender.find_title_matches(ratings, search))
    assert all(search.lower() in title.lower() for title in matches)
    if search in ("(GOTY", "(", "3: Wild Hunt ("):
        assert matches == ["The Witcher 3: Wild Hunt (GOTY)"]


def test_small_catalog_recommends_every_other_game(small_catalog):
    # Three games in all, so only two candidates exist.
    assert small_catalog.X.shape[0] == 3

    recommendations = small_catalog.recommend("Stardew Valley")

    assert sorted(recommendations) == ["Starbound", "Terraria"]
