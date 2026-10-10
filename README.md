# Steam Recommender System

A command-line game recommender built on Steam review data. Type in a game you like and it suggests five others that the same players tended to recommend.

```
Enter a game title (or 'quit' to exit): Stardew Valley

Because you liked **Stardew Valley**, you might also enjoy:
- Hades
- Terraria
- Don't Starve Together
- Starbound
- Portal 2
```

## How it works

This is item-based collaborative filtering, done with linear algebra:

1. Every review becomes an entry in a sparse **game × user matrix**. A cell is 1 if that user recommended that game, and 0 otherwise.
2. Each game is then a row vector with one dimension per user (about 13.8 million of them).
3. For the game you enter, the program finds the five rows with the smallest **cosine distance** to it, using scikit-learn's `NearestNeighbors`.

Two games end up close together when largely the same set of players recommended both.

## Dataset

The data is not included in this repository because it is too large for GitHub. Download [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) from Kaggle and unzip it into the `Linear algebra` folder, next to `Linear_algebra.py`. The script reads:

| File | Size | Contents |
| --- | --- | --- |
| `recommendations.csv` | 1.9 GB | About 41 million user reviews |
| `games.csv` | 4.6 MB | Game titles and store details |
| `games_metadata.json` | 16.7 MB | Descriptions and tags |

## Setup

Requires Python 3. From the repository root:

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install numpy pandas scipy scikit-learn
```

`py` is the Python launcher that the python.org installer adds on Windows. On other systems use `python3` and `.venv/bin/python`.

## Running

```powershell
cd "Linear algebra"
..\.venv\Scripts\python.exe Linear_algebra.py
```

Loading the reviews takes about three minutes before the prompt appears. After that:

- Enter an exact title to get recommendations.
- Enter part of a title (for example `witcher`) to list the matching games.
- Enter `quit` to exit.

The project can also be opened in Visual Studio through `Linear algebra.sln`.

## Tests

The `tests` folder runs the whole pipeline on a small made-up dataset (10 games, 28 reviews), so the tests need neither the Kaggle download nor the three-minute load:

```powershell
.venv\Scripts\python.exe -m pip install pytest
.venv\Scripts\python.exe -m pytest tests -v
```

[TESTING.md](TESTING.md) describes the dataset, what each test checks, the latest results, and the three bugs the tests found.

## Team

- [Nathan Chiamsachang](https://github.com/nchiamsachang)
- [Trey Rajsombath](https://github.com/TreyRajsombath)
