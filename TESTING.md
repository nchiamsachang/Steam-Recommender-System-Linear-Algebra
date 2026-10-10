# Tests

The tests run the real recommendation pipeline (`load_data` → `attach_titles` →
`create_matrix` → `recommend_similar`, plus the title search) on a tiny synthetic dataset.
Nothing is mocked, the 1.9 GB Kaggle files are not read, and no network access is used.

## Running them

From the repository root, with the virtual environment from `README.md`:

```powershell
.venv\Scripts\python.exe -m pip install pytest      # once
.venv\Scripts\python.exe -m pytest tests -v
```

## The synthetic dataset (`tests/data/`)

Three files with the same names and columns as the Kaggle ones: 10 games, 28 reviews,
10 users. Which users recommended which game:

| Game | Recommended by users | Why it is there |
|---|---|---|
| Portal | 1 2 3 4 (user 9 reviewed it with a thumbs-down) | |
| Portal 2 | 1 2 3 4 | Identical to Portal: a tie at distance 0 |
| Half-Life 2 | 1 2 3 | |
| Stardew Valley | 5 6 7 8 (user 2: thumbs-down) | The "known game" |
| Terraria | 5 6 7 | Similarity to Stardew Valley 0.87 |
| Starbound | 5 6 | 0.71 |
| The Witcher 3: Wild Hunt (GOTY) | 1 8 | 0.35; a title with brackets |
| Hades | 3 4 5 | 0.29 |
| Game Without Metadata | 10 | Missing from `games_metadata.json`, so it is dropped on load |
| Game Nobody Reviewed | — | No reviews, so it never reaches the matrix |

Similarity is the cosine between two games' rows: users who recommended both, divided by
the square root of (recommenders of one × recommenders of the other).

## What is tested (`tests/test_recommender.py`)

| Test | Checks |
|---|---|
| `test_matrix_has_one_row_per_reviewed_game_and_one_column_per_reviewer` | The matrix is 8 games × 9 users, with 25 ones (27 reviews, 2 of them thumbs-down) |
| `test_known_game_gets_five_recommendations` | Stardew Valley gets 5 results, including all four games that share recommenders with it |
| `test_recommendations_leave_out_the_queried_game_and_repeat_nothing` | For every game: 5 results, none repeated, never the game itself |
| `test_a_game_with_identical_reviewers_is_recommended_not_mistaken_for_the_query` | Portal recommends Portal 2 and the other way round |
| `test_missing_title_gives_no_recommendations` | An unknown title returns nothing and prints "not found" |
| `test_partial_title_search_finds_the_expected_games` (6 inputs) | Part of a title, in any case, finds the right games; no match finds none |
| `test_search_text_is_matched_literally` (6 inputs) | `(`, `[`, `+`, `\` and similar are searched for as typed |
| `test_small_catalog_recommends_every_other_game` | With 3 games in the catalog, the other 2 are returned |

## Results, 2026-10-09

Commit `62fbc2f` plus uncommitted changes. Windows 11, Python 3.13.1, pytest 9.1.1,
numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1.

| Code under test | Result |
|---|---|
| `62fbc2f` with only the loading and search code moved into functions (no behaviour change) | 9 passed, **9 failed** |
| The same with the three fixes below | **18 passed**, 0 failed |

pytest counts each input of a parametrized test separately: the 18 are 8 test functions.

### Bugs the tests found (all three fixed)

| # | What happened | Cause | Fix |
|---|---|---|---|
| 1 | With fewer than 6 games in the catalog, asking for recommendations stopped the program with `ValueError: Expected n_neighbors <= n_samples_fit`. | The code always asks for k + 1 neighbours. | Ask for `min(k + 1, number of games)`. |
| 2 | Entering "Portal 2" recommended "Portal 2", and left out "Portal". | The code drops the first neighbour, assuming it is the queried game. When another game has exactly the same recommenders both are at distance 0 and either can come first. | Remove the queried game by its index, then keep the first k. |
| 3 | Typing `(`, `[`, `+` or `\` at the prompt stopped the program with `re.PatternError`. | The partial-title search treated the text as a regular expression. | Search for the text literally (`regex=False`). |

Bug 2 needs two games with identical review vectors. That was produced in the synthetic
data; how often it happens in the real dataset was not measured.

### Changes to `Linear_algebra.py`

- `load_data(data_dir=".")`, `attach_titles(...)` and `find_title_matches(...)` were split
  out of `main()` so that tests can point the loader at `tests/data`. `main()` calls them
  with the same files, in the same order, and prints the same messages.
- The three fixes above.

### Check on the real dataset

The changed program was also run once on the full Kaggle files, with input from a file
(`Stardew Valley`, `Portal 2`, `witcher`, `quit`). This is a manual check, not part of the
test suite.

- It loaded 37,518 games and 13,781,059 users and exited normally (223 s in all).
- `Stardew Valley` gave Hades, Terraria, Don't Starve Together, Starbound and Portal 2:
  the same five titles, in the same order, as the sample output in `README.md`.
- `Portal 2` gave five other games and not itself.
- `witcher` listed 6 matching titles.
- In a separate run, typing `(` listed 318 matching titles; before fix 3 it stopped the
  program.

## Behaviour that is ambiguous and deliberately not asserted

These are how the program behaves today. The tests do not lock them in, because the
repository does not say what is intended.

- **Order of the results.** Recommendations are printed in the order the games first
  appear in the review file, not from most to least similar. On the synthetic data,
  Half-Life 2 (similarity 0) is listed before Terraria (0.87) for Stardew Valley. The tests
  compare sets, not order.
- **Ties for the last place.** When several games are equally similar, which of them makes
  the list is decided by scikit-learn's ordering. The Stardew Valley test accepts any of
  the three tied games in fifth place.
- **Two different games with the same title** are shown once, so fewer than five titles can
  be printed. Not tested.
- **A game nobody recommended** (only thumbs-down reviews) is an all-zero row. Not tested.
- **Exact titles are case-sensitive, partial ones are not.** `stardew valley` goes to the
  partial search and still finds the one match. Not asserted either way.
- **Thumbs-down reviews** only add a zero to the matrix, so "disliked by the same people"
  does not make games similar. The dimension test records this (users 2 and 9 are columns)
  without judging it.
- **Return type.** A missing title returns an empty list; a found one returns an array.
  The tests convert both to a list.

## Not done

- Coverage was not measured (no coverage package is installed).
- Recommendation quality on the real dataset is not tested: the tests check the mechanics,
  not whether the suggestions are good.
