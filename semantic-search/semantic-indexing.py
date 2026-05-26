import json
import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

#
# IMPORTANTE
#
# Añadir manualmente el archivo "questions_with_embeddings.ndjson"
# en la carpeta embeddings_quora
#

MODEL_NAME = "google/embeddinggemma-300m"
HF_TOKEN = "changeme"
BATCH_SIZE = 5000  # must be below Chroma's max batch size

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DUPLICATE_THRESHOLD = 0.25

# ------------------------- Duplicate detection ------------------------- #

def prepare_model():
    """
    Prepara el modelo de embeddings, carga los datos y busca duplicados.

    Este método inicializa el modelo de `SentenceTransformer`, crea o carga una
    colección en ChromaDB, inserta los registros si aún no existen y finalmente
    ejecuta la búsqueda de duplicados.

    Returns:
        set: Conjunto de pares de identificadores considerados duplicados.
    """
    print("Preparing model...")
    model = SentenceTransformer(MODEL_NAME, token=HF_TOKEN)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)

    client = chromadb.PersistentClient(path="./chroma_index")

    collection = client.get_or_create_collection(
        name="quora",
        embedding_function=embedding_fn
    )

    records = []

    json_path = os.path.join(SCRIPT_DIR,"embeddings_quora", "questions_with_embeddings.ndjson")

    with open(json_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading records"):
            records.append(json.loads(line))

    if collection.count() > 0:
        print(f"Collection already contains {collection.count()} documents. Skipping insertion.")
    else:
        batch_insertion(collection, records)

    return process_queries(collection, records)


def batch_insertion(collection, records):
    """
    Inserta los registros en la colección de ChromaDB en lotes.

    Args:
        collection (chromadb.api.models.Collection.Collection): Colección Chroma donde se insertarán los datos.
        records (list): Lista de diccionarios con las claves 'id', 'text' y 'embedding'.
    """
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i+BATCH_SIZE]
        ids = [str(r["id"]) for r in batch]
        texts = [r["text"] for r in batch]
        embeddings = [r["embedding"] for r in batch]
        collection.add(ids=ids, documents=texts, embeddings=embeddings)
        print(f"Indexed batch {i}-{i+len(batch)-1}")    


def process_queries(collection, records):
    """
    Procesa las consultas de los registros para encontrar duplicados.

    Busca los elementos más similares en la colección y considera duplicados
    aquellos cuya distancia esté por debajo del umbral `DUPLICATE_THRESHOLD`.

    Args:
        collection (chromadb.api.models.Collection.Collection): Colección con los documentos indexados.
        records (list): Lista de registros con identificador, texto y embedding.

    Returns:
        set: Conjunto de pares de identificadores de preguntas duplicadas.
    """
    dup_set = set()
    dup_count = 0
    avg_distance = 0
    for record in tqdm(records, desc="Looking for duplicates..."):
        query_text = record["text"]
        query_id = str(record["id"])
        
        results = collection.query(query_texts=[query_text], n_results=10)
        for doc_id, doc_text, dist in zip(results["ids"][0], results["documents"][0], results["distances"][0]):
            if doc_id != query_id and dist < DUPLICATE_THRESHOLD:
                avg_distance += dist
                pair = tuple(sorted([query_id, doc_id]))
                if pair not in dup_set:
                    dup_set.add(pair)
                    dup_count += 1

    avg_distance = avg_distance/dup_count
    print(f"Number of duplicates found: {dup_count}")  
    print(f"Average distance between duplicates: {avg_distance}") 
    return dup_set

# ------------------------- Reading truths ------------------------- #

def parse_truths(file_name):
    """
    Parsea el archivo de verdades para obtener pares de identificadores duplicados.

    Cada línea del archivo debe contener dos identificadores separados por un espacio.
    
    Args:
        file_name (str): Ruta del archivo de verdades.
    
    Returns:
        set: Conjunto de pares de identificadores (ordenados).
    """
    truth_pairs = set()
    with open(file_name, "r") as t:
        for line in tqdm(t.readlines(), desc="Loading truths..."):
            t1 = line.split()[0].strip()
            t2 = line.split()[1].strip()
            pair = tuple(sorted([t1, t2]))
            truth_pairs.add(pair)
    print("Truths loaded.")
    return truth_pairs

# ------------------------- Computing values ------------------------- #
def compute_values(results, truths):
    """
    Calcula VP (verdaderos positivos), FP (falsos positivos) y FN (falsos negativos).
    
    Args:
        results (set): Pares de duplicados encontrados en ChromaDB
        truths (set): Pares verdaderos del archivo de verdades.
    
    Returns:
        tuple: VP, FP, FN.
    """
    checked_truths = set()
    VP = 0
    FP = 0
    FN = 0
    for tup in tqdm(results, desc="Computing VP, FP and FN..."):
        if tup in truths:
            VP += 1
            checked_truths.add(tup)
        else:
            FP += 1
    if len(checked_truths) < len(truths):
        FN = len(truths) - len(checked_truths)
    
    return VP, FP, FN

def compute_precision_recall(VP, FP, FN):
    """
    Calcula precisión y exhaustividad (recall) a partir de VP, FP y FN.
    
    Args:
        VP (int): Verdaderos positivos.
        FP (int): Falsos positivos.
        FN (int): Falsos negativos.
    
    Returns:
        tuple: (precision, recall)
            - precision (float): Proporción de predicciones correctas sobre el total de predicciones positivas.
            - recall (float): Proporción de verdaderos positivos detectados sobre el total real de positivos.
    """
    precision = VP / (VP + FP)
    recall = VP / (VP + FN)
    return precision, recall

def compute_F1(precision, recall):
    """
    Calcula la métrica F1 a partir de precisión y exhaustividad.
    
    Args:
        precision (float): Precisión.
        recall (float): Exhaustividad.
    
    Returns:
        float: Valor F1.
    """
    return 2 * (precision * recall) / (precision + recall)

# ------------------------- Execution ------------------------- #

quora_path = os.path.join(SCRIPT_DIR,"quora", "duplicates.txt")
truths = parse_truths(quora_path)
dup_set = prepare_model()
VP, FP, FN = compute_values(dup_set, truths)
P, R = compute_precision_recall(VP, FP, FN)
F1 = compute_F1(P, R)

print("\n------------------------------\nRESULTS FOR QUORA COLLECTION")
print(f"VP = {VP}")
print(f"FP = {FP}")
print(f"FN = {FN}")
print(f"Precision = {P}")
print(f"Recall = {R}")
print(f"F1 = {F1}")
