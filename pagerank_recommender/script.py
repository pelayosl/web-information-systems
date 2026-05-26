import pandas as pd
from itertools import combinations
from collections import defaultdict
import igraph as ig
from tqdm import tqdm

RATINGS = pd.read_csv("files/ratings.csv", usecols=["userId", "movieId", "rating"])

def generate_movie_dict():
    print("🔄 Generando diccionario de co-ocurrencias de películas...")

    movie_pairs = defaultdict(int)
    user_groups = RATINGS.groupby("userId")["movieId"].apply(list)

    for movies in tqdm(user_groups, desc="Procesando usuarios", unit="usuario"):
        for pair in combinations(sorted(set(movies)), 2):
            movie_pairs[pair] += 1

    print(f"✅ Diccionario generado con {len(movie_pairs)} pares de películas.")

    return movie_pairs
    # { (1, 2): 3, (1, 3): 5, (2, 4): 1, ... }

def generate_movie_graph(movie_pairs):
    print("🔄 Construyendo grafo global de películas...")

    edges = list(movie_pairs.keys())
    weights = list(movie_pairs.values())

    g = ig.Graph()

    unique_movies = sorted(set(RATINGS["movieId"]))

    # Un vértice por cada película
    g.add_vertices(len(unique_movies))
    # Asignar el atributo "movieId" a cada vértice
    g.vs["movieId"] = unique_movies

    # Convertir los pares de movieId en pares de índices
    id_to_index = {mid: idx for idx, mid in enumerate(unique_movies)}
    edge_indices = [(id_to_index[a], id_to_index[b]) for a, b in edges]

    g.add_edges(edge_indices)
    g.es["weight"] = weights
    print("✅ Grafo creado correctamente.")
    return g

def compute_global_pagerank(g):
    print("🔄 Calculando PageRank global de películas...")
    pagerank_scores = g.pagerank(weights=g.es["weight"])

    ranking_df = pd.DataFrame({
    "movieId": g.vs["movieId"],
    "pagerank": pagerank_scores
    }).sort_values(by="pagerank", ascending=False)

    movies = pd.read_csv("files/movies.csv")
    ranking_df = ranking_df.merge(movies, on="movieId", how="left")

    ranking_df.to_csv("movie_global_ranking.txt", index=False,
                  columns=["movieId", "title", "pagerank"])
    print("✅ Archivo 'movie_global_ranking.txt' generado correctamente.")


if __name__ == "__main__":
    movie_pairs = generate_movie_dict()
    g = generate_movie_graph(movie_pairs)
    compute_global_pagerank(g)

