import pandas as pd
import os
from itertools import combinations
from collections import defaultdict
import igraph as ig
from tqdm import tqdm
from utils import jaccard, string_to_bag_of_words, safe_filename

script_dir = os.path.dirname(os.path.abspath(__file__))
ratings_csv_path = os.path.join(script_dir, "files", "ratings.csv")
movies_csv_path = os.path.join(script_dir, "files", "movies.csv")

RATINGS = pd.read_csv(ratings_csv_path, usecols=["userId", "movieId", "rating"])
MOVIES = pd.read_csv(movies_csv_path, usecols=["movieId", "title", "genres"])

def get_movie_by_title(query_title):
    """
    Busca una película por título en el DataFrame MOVIES.
    Si hay varias coincidencias, solicita ser más específico.
    Si no hay coincidencias, informa al usuario.
    Args:
        query_title (str): Título o parte del título de la película a buscar.
    Returns:
        tuple: (movieId, title) si se encuentra una coincidencia única, None en caso contrario.
    """
    matches = MOVIES[MOVIES["title"].str.contains(query_title, case=False, na=False, regex=False)]
    if matches.empty:
        print(f"❌ No se encontró ninguna película que coincida con '{query_title}'.")
        return None
    
    if len(matches) > 1:
        print("⚠️ Se encontraron múltiples coincidencias, sé más específico:")
        print(matches[["movieId", "title"]].to_string(index=False))
        return None
    
    movie = matches.iloc[0]
    print(f"✅ Película encontrada: {movie['title']} (movieId={movie['movieId']})")
    return movie["movieId"], movie["title"]

def build_local_graph(movie_id):
    """
    Construye un grafo local considerando solo los usuarios que han visto la película indicada.
    Los nodos son películas vistas por estos usuarios y las aristas representan co-ocurrencias.
    Args:
        movie_id (int): ID de la película base.
    Returns:
        tuple: (grafo igraph, vector de personalización para PageRank)
    """
    print(f"Construyendo grafo local para la película {movie_id}...")

    users_who_watched = RATINGS.loc[RATINGS["movieId"] == movie_id, "userId"].unique()
    local_ratings = RATINGS[RATINGS["userId"].isin(users_who_watched)]
    mean_ratings = (
        local_ratings.groupby("movieId")["rating"]
        .mean()
        .to_dict()
    )

    filtered_movie_ids = sorted(local_ratings["movieId"].unique())

    # Filtrar las películas teniendo en cuenta el género con JACCARD
    filtered_movie_ids, genre_jaccard_map = build_jaccard_map(filtered_movie_ids)

    # Generar co-ocurrencias SOLO entre esas películas
    movie_pairs = defaultdict(int)
    user_groups = local_ratings.groupby("userId")["movieId"].apply(list)

    for movies in tqdm(user_groups, desc="Procesando usuarios (local)", unit="usuario"):
        for pair in combinations(sorted(set(movies)), 2):
            movie_pairs[pair] += 1

    # Grafo local
    g = ig.Graph()
    g.add_vertices(len(filtered_movie_ids))
    g.vs["movieId"] = filtered_movie_ids
    g.vs["genre_jaccard"] = [genre_jaccard_map[mid] for mid in filtered_movie_ids]

    personalization = build_personalization(filtered_movie_ids, mean_ratings)

    id_to_index = {mid: idx for idx, mid in enumerate(filtered_movie_ids)}
    edges_filtered = []
    weights_filtered = []

    for (a, b), w in movie_pairs.items():
        if a in id_to_index and b in id_to_index:
            edges_filtered.append((id_to_index[a], id_to_index[b]))
            weights_filtered.append(w)

    g.add_edges(edges_filtered)
    g.es["weight"] = weights_filtered

    print(f"✅ Grafo local: {len(filtered_movie_ids)} nodos, {len(edges_filtered)} aristas.")
    return g, personalization

def build_jaccard_map(filtered_movie_ids):
    """
    Calcula el coeficiente de Jaccard entre los géneros de la película base y las demás películas filtradas.
    Args:
        filtered_movie_ids (list): Lista de IDs de películas filtradas.
    Returns:
        tuple: (lista ordenada de IDs por similitud, diccionario movieId->jaccard)
    """
    base_genres_series = MOVIES.loc[MOVIES["movieId"] == movie_id, "genres"]
    base_bag = string_to_bag_of_words(base_genres_series.iloc[0])

    genre_jaccard_map = {}
    for mid in filtered_movie_ids:
        genres_series = MOVIES.loc[MOVIES["movieId"] == mid, "genres"]
        bag = string_to_bag_of_words(genres_series.iloc[0])
        try:
            score = jaccard(base_bag, bag)
        except Exception:
            score = 0.0
        genre_jaccard_map[mid] = score

    return sorted(filtered_movie_ids, key=lambda m: genre_jaccard_map.get(m, 0.0), reverse=True), genre_jaccard_map

def build_personalization(filtered_movie_ids, mean_ratings):
    """
    Construye el vector de personalización para PageRank normalizando las valoraciones medias.
    Args:
        filtered_movie_ids (list): Lista de IDs de películas filtradas.
        mean_ratings (dict): Diccionario movieId->media de valoraciones.
    Returns:
        list: Vector de personalización normalizado.
    """
    min_rating = min(mean_ratings.values())
    max_rating = max(mean_ratings.values())

    personalization = []
    for mid in filtered_movie_ids:
        rating = mean_ratings.get(mid, 0.0)
        normalized = (rating - min_rating) / (max_rating - min_rating)
        personalization.append(normalized)
    return personalization

def compute_local_pagerank(g, base_movie_title, personalization, alpha=0.7):
    """
    Calcula el PageRank personalizado sobre el grafo local y genera un archivo de recomendaciones.
    Args:
        g (igraph.Graph): Grafo local de películas.
        base_movie_title (str): Título de la película base.
        personalization (list): Vector de personalización para PageRank.
        alpha (float): Peso relativo entre PageRank y similitud de género.
    """
    print("Calculando PageRank local...")
    pagerank_scores = g.personalized_pagerank(
        directed=True,
        damping=0.85,
        weights=g.es["weight"],
        reset=personalization
    )

    ranking_df = pd.DataFrame({
        "movieId": g.vs["movieId"],
        "pagerank": pagerank_scores,
        "genre_jaccard": g.vs["genre_jaccard"]
    })

    ranking_df["pagerank_norm"] = ranking_df["pagerank"] / ranking_df["pagerank"].max()

    genre_adjusted_score = alpha * ranking_df["pagerank_norm"] + (1 - alpha) * ranking_df["genre_jaccard"]
    ranking_df["final_score"] = (genre_adjusted_score)

    ranking_df = ranking_df.sort_values(by="final_score", ascending=False).head(10)
    ranking_df = ranking_df.merge(MOVIES, on="movieId", how="left")

    output_name = f"recommendations_{safe_filename(base_movie_title)}.txt"
    ranking_df.to_csv(output_name, index=False, columns=["movieId", "title", "genres", "pagerank_norm","final_score"])

    print(f"✅ Archivo '{output_name}' generado correctamente.")

if __name__ == "__main__":
    query = input("Introduce el título (o parte del título) de la película: ").strip()
    result = get_movie_by_title(query)

    while result is None:
        query = input("Introduce el título (o parte del título) de la película: ").strip()
        result = get_movie_by_title(query)

    movie_id, title = result

    g_local, personalization = build_local_graph(movie_id)
    compute_local_pagerank(g_local, title, personalization, alpha=0.7) 

