import pandas as pd
import numpy as np
import os
import time

# ─────────────────────────────────────────────
# CONFIG
# Yahan hum Netflix ke raw files parse karte hain
# combined_data files mein format hai:
#   MovieID:
#   UserID,Rating,Date
# ─────────────────────────────────────────────
DATA_DIR   = r"C:\Users\vanqu\OneDrive\Desktop\fm_replication\data"
OUTPUT_FILE = os.path.join(DATA_DIR, "ratings_sample.csv")

# 2 MILLION ratings — paper ke results ke karib aane ke liye
MAX_RATINGS = 2_000_000

COMBINED_FILES = [
    os.path.join(DATA_DIR, "combined_data_1.txt"),
    os.path.join(DATA_DIR, "combined_data_2.txt"),
    os.path.join(DATA_DIR, "combined_data_3.txt"),
    os.path.join(DATA_DIR, "combined_data_4.txt"),
]
MOVIE_TITLES_FILE = os.path.join(DATA_DIR, "movie_titles.csv")


def parse_netflix_files(file_list, max_ratings=2_000_000):
    """
    Netflix data parse karta hai
    Format: MovieID: phir UserID,Rating,Date
    """
    records = []
    current_movie_id = None
    total = 0

    print(f"Netflix files parse ho rahi hain...")
    print(f"Target: {max_ratings:,} ratings\n")
    start = time.time()

    for filepath in file_list:
        if total >= max_ratings:
            break
        print(f"  Reading: {os.path.basename(filepath)}")
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.endswith(":"):
                    current_movie_id = int(line[:-1])
                    continue
                parts = line.split(",")
                if len(parts) == 3:
                    records.append({
                        "movie_id": current_movie_id,
                        "user_id":  int(parts[0]),
                        "rating":   int(parts[1]),
                        "date":     parts[2]
                    })
                    total += 1
                    if total >= max_ratings:
                        break
        print(f"    Collected: {total:,} ratings")

    elapsed = time.time() - start
    print(f"\nParsing done in {elapsed:.1f}s")
    print(f"Total ratings: {total:,}")
    return pd.DataFrame(records)


def load_movie_titles(filepath):
    movies = []
    with open(filepath, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split(",", 2)
            if len(parts) == 3:
                movies.append({
                    "movie_id": int(parts[0]),
                    "year":     parts[1],
                    "title":    parts[2]
                })
    return pd.DataFrame(movies)


def run_eda(ratings_df, movies_df):
    """
    EDA = Exploratory Data Analysis
    Data ko samajhte hain: kitne users, movies, sparsity
    """
    print("\n" + "="*50)
    print("  EXPLORATORY DATA ANALYSIS")
    print("="*50)

    n_users   = ratings_df["user_id"].nunique()
    n_movies  = ratings_df["movie_id"].nunique()
    n_ratings = len(ratings_df)

    # SPARSITY = kitne ratings MISSING hain
    # Agar 1000 users, 100 movies → 100,000 possible pairs
    # Agar sirf 5000 ratings hain → 95% empty
    # YEH HI PROBLEM HAI JO FM SOLVE KARTA HAI
    sparsity = 1 - (n_ratings / (n_users * n_movies))

    print(f"\nRatings shape     : {ratings_df.shape}")
    print(f"Unique users      : {n_users:,}")
    print(f"Unique movies     : {n_movies:,}")
    print(f"Rating range      : "
          f"{ratings_df['rating'].min()} - "
          f"{ratings_df['rating'].max()}")
    print(f"Mean rating       : "
          f"{ratings_df['rating'].mean():.4f}")
    print(f"\nSPARSITY          : {sparsity:.6f}")
    print(f"  ({sparsity*100:.4f}% ratings missing)")
    print(f"  --> Yahi reason hai ki SVM fail hota hai!")
    print(f"  --> FM is problem ko solve karta hai!")

    print(f"\nRating Distribution:")
    dist = ratings_df["rating"].value_counts().sort_index()
    for r, c in dist.items():
        bar = "+" * int(c / n_ratings * 40)
        print(f"  {r} star : {bar} "
              f"{c:,} ({c/n_ratings*100:.1f}%)")

    # Top movies
    movie_stats = (
        ratings_df.groupby("movie_id")["rating"]
        .agg(["count","mean"])
        .reset_index()
        .rename(columns={
            "count":"n_ratings",
            "mean":"avg_rating"
        })
    )
    movie_stats = movie_stats[
        movie_stats["n_ratings"] >= 100
    ]
    top = (
        movie_stats
        .sort_values("avg_rating", ascending=False)
        .head(5)
        .merge(movies_df[["movie_id","title"]],
               on="movie_id", how="left")
    )
    print(f"\nTop 5 Movies (min 100 ratings):")
    for _, row in top.iterrows():
        print(f"  {str(row['title'])[:35]:<35} | "
              f"{row['avg_rating']:.2f} stars | "
              f"{int(row['n_ratings']):,} ratings")

    user_counts = ratings_df.groupby(
        "user_id")["rating"].count()
    print(f"\nUser Activity:")
    print(f"  Avg ratings/user : {user_counts.mean():.1f}")
    print(f"  Max ratings/user : {user_counts.max():,}")

    return n_users, n_movies, sparsity


def filter_and_save(ratings_df, movies_df,
                    output_path,
                    min_user_ratings=10,
                    min_movie_ratings=20):
    """
    Bahut sparse users/movies hatate hain
    Jo user sirf 1-2 ratings deta hai use
    hata dete hain — FM usse kuch seekh nahi sakta
    """
    print(f"\nFiltering sparse users/movies...")
    print(f"  Before: {len(ratings_df):,} ratings | "
          f"{ratings_df['user_id'].nunique():,} users | "
          f"{ratings_df['movie_id'].nunique():,} movies")

    for _ in range(2):
        u = ratings_df.groupby(
            "user_id")["rating"].count()
        m = ratings_df.groupby(
            "movie_id")["rating"].count()
        ratings_df = ratings_df[
            ratings_df["user_id"].isin(
                u[u >= min_user_ratings].index) &
            ratings_df["movie_id"].isin(
                m[m >= min_movie_ratings].index)
        ]

    print(f"  After : {len(ratings_df):,} ratings | "
          f"{ratings_df['user_id'].nunique():,} users | "
          f"{ratings_df['movie_id'].nunique():,} movies")

    # Re-index from 0 (FM ke liye zaruri)
    user_map  = {u: i for i, u in enumerate(
        sorted(ratings_df["user_id"].unique()))}
    movie_map = {m: i for i, m in enumerate(
        sorted(ratings_df["movie_id"].unique()))}

    ratings_df = ratings_df.copy()
    ratings_df["user_idx"]  = ratings_df[
        "user_id"].map(user_map)
    ratings_df["movie_idx"] = ratings_df[
        "movie_id"].map(movie_map)

    final_df = ratings_df.merge(
        movies_df[["movie_id","title","year"]],
        on="movie_id", how="left"
    )
    final_df.to_csv(output_path, index=False)

    np.save(os.path.join(DATA_DIR,"user_mapping.npy"),
            user_map, allow_pickle=True)
    np.save(os.path.join(DATA_DIR,"movie_mapping.npy"),
            movie_map, allow_pickle=True)

    print(f"\nSaved: {output_path}")
    return final_df, len(user_map), len(movie_map)


if __name__ == "__main__":
    ratings_df = parse_netflix_files(
        COMBINED_FILES, max_ratings=MAX_RATINGS)
    movies_df  = load_movie_titles(MOVIE_TITLES_FILE)

    n_users, n_movies, sparsity = run_eda(
        ratings_df, movies_df)

    final_df, n_u, n_m = filter_and_save(
        ratings_df, movies_df,
        output_path=OUTPUT_FILE,
        min_user_ratings=10,
        min_movie_ratings=20
    )

    print(f"\n{'='*50}")
    print(f"  SUMMARY")
    print(f"{'='*50}")
    print(f"  n_users  |U| : {n_u:,}")
    print(f"  n_movies |I| : {n_m:,}")
    print(f"  Feature size : {n_u + n_m:,}")
    print(f"  Sparsity     : {sparsity*100:.4f}%")
    print(f"\n  Next: python feature_engineering.py")