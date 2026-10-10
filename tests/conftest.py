"""Shared fixtures: the recommender module and the tiny synthetic dataset in tests/data.

The dataset stands in for the 1.9 GB Kaggle files. It is read through the same
load_data -> attach_titles -> create_matrix steps as the real one, so nothing about the
pipeline is mocked.

Who recommended what (user ids):

    Portal                            1 2 3 4      (user 9 reviewed it but did not recommend it)
    Portal 2                          1 2 3 4      (identical to Portal)
    Half-Life 2                       1 2 3
    Stardew Valley                    5 6 7 8      (user 2 reviewed it but did not recommend it)
    Terraria                          5 6 7
    Starbound                         5 6
    Hades                             3 4 5
    The Witcher 3: Wild Hunt (GOTY)   1 8

Two more games are there to be left out: "Game Without Metadata" (reviewed by user 10, but
missing from games_metadata.json) and "Game Nobody Reviewed".
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "tests", "data")
sys.path.insert(0, os.path.join(REPO_ROOT, "Linear algebra"))

import Linear_algebra as recommender  # noqa: E402


class Catalog:
    """A loaded dataset and its matrix, with recommend() bound to them."""

    def __init__(self, ratings):
        self.ratings = ratings
        self.X, self.game_mapper, self.game_inv_mapper = recommender.create_matrix(ratings)

    def recommend(self, title, k=5):
        return list(recommender.recommend_similar(title, self.ratings, self.X, self.game_mapper,
                                                  self.game_inv_mapper, k=k))


@pytest.fixture(scope="session")
def ratings():
    return recommender.attach_titles(*recommender.load_data(DATA_DIR))


@pytest.fixture
def catalog(ratings):
    return Catalog(ratings)


@pytest.fixture
def small_catalog(ratings):
    """Only three games, so fewer than five candidates exist."""
    return Catalog(ratings[ratings["title"].isin(["Stardew Valley", "Terraria", "Starbound"])])
