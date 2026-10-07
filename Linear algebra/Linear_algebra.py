
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.neighbors import NearestNeighbors

def create_matrix(df):
    """Create a sparse matrix for collaborative filtering"""
    user_mapper = {uid: i for i, uid in enumerate(df['user_id'].unique())}
    game_mapper = {mid: i for i, mid in enumerate(df['app_id'].unique())}
    game_inv_mapper = {i: mid for mid, i in game_mapper.items()}

    user_index = df['user_id'].map(user_mapper)
    game_index = df['app_id'].map(game_mapper)

    X = scipy.sparse.csr_matrix(
        (df["is_recommended"], (game_index, user_index)),
        shape=(len(game_mapper), len(user_mapper))
    )
    return X, game_mapper, game_inv_mapper

def recommend_similar(game_title, df, X, game_mapper, game_inv_mapper, k=5):
    """Find similar games based on user preferences"""
    try:
        game_id = df[df['title'] == game_title]['app_id'].iloc[0]
        game_idx = game_mapper[game_id]
        game_vec = X[game_idx]

        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(X)
        distances, indices = model.kneighbors(game_vec, n_neighbors=k + 1)

        neighbor_ids = [game_inv_mapper[i] for i in indices.flatten()[1:]]
        recommendations = df[df['app_id'].isin(neighbor_ids)]['title'].unique()

        print(f"\nBecause you liked **{game_title}**, you might also enjoy:")
        for rec in recommendations:
            print(f"- {rec}")
        
        return recommendations
    except IndexError:
        print(f"Game '{game_title}' not found in the dataset.")
        return []

def main():
    print("Loading data...")
    
    # Load the CSV files
    # Make sure these files are in the same directory as this script
    try:
        ratings = pd.read_csv("recommendations.csv")
        games = pd.read_csv("games.csv")
        users = pd.read_csv("users.csv")
        game_meta = pd.read_json("games_metadata.json", lines=True)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure all CSV and JSON files are in the same directory as this script.")
        return

    # Merge game metadata
    print("Processing data...")
    games = pd.merge(games, game_meta, on='app_id', how='inner')
    ratings = pd.merge(ratings, games[['app_id', 'title']], on='app_id', how='inner')

    # Create the recommendation matrix
    X, game_mapper, game_inv_mapper = create_matrix(ratings)
    
    print(f"\nLoaded {len(ratings['title'].unique())} games and {len(ratings['user_id'].unique())} users")
    print("\n" + "="*50)
    print("GAME RECOMMENDATION SYSTEM")
    print("="*50)

    # Interactive loop
    while True:
        game_search = input("\nEnter a game title (or 'quit' to exit): ").strip()
        
        if game_search.lower() == 'quit':
            print("Thanks for using the recommendation system!")
            break
        
        if not game_search:
            continue
        
        # Check if exact match exists
        if game_search in ratings['title'].values:
            recommend_similar(game_search, ratings, X, game_mapper, game_inv_mapper, k=5)
        else:
            # Try to find partial matches
            matches = ratings[ratings['title'].str.contains(game_search, case=False, na=False)]['title'].unique()
            
            if len(matches) == 0:
                print(f"No games found matching '{game_search}'")
            elif len(matches) == 1:
                recommend_similar(matches[0], ratings, X, game_mapper, game_inv_mapper, k=5)
            else:
                print(f"\nFound {len(matches)} games matching '{game_search}':")
                for i, match in enumerate(matches[:10], 1):
                    print(f"{i}. {match}")
                if len(matches) > 10:
                    print(f"... and {len(matches) - 10} more")
                print("\nPlease enter the exact game title to get recommendations.")

if __name__ == "__main__":
    main()