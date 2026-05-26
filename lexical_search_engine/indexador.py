import json
import math
import pickle
import os
from collections import defaultdict
from tqdm import tqdm
from .utils import tokenizar_texto

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(PROJECT_ROOT, "buscador_lexico", "datos", "LISA-collection-JSON", "lisa-corpus.json")
INDICE_PATH = os.path.join(PROJECT_ROOT, "buscador_lexico", "modelos", "indice_lisa.pkl")

def cargar_documentos_lisa(ruta=CORPUS_PATH):
    """
    Carga la colección de documentos LISA desde un archivo JSON.

    Parámetros:
        ruta (str): Ruta del archivo JSON que contiene los documentos.

    Retorna:
        list: Lista de diccionarios, cada uno representando un documento con sus campos.
    """
    with open(ruta, "r", encoding="utf-8") as f:
        documentos = json.load(f)
    for doc in documentos:
        doc["content"] = doc["content"].lower()
        doc["title"] = doc["title"].lower()
    return documentos

def generar_indice(documentos, stopwords):
    """
    Genera un índice invertido con ponderaciones TF e IDF para la colección LISA.

    Parámetros:
        documentos (list): Lista de documentos, cada uno con 'id', 'title' y 'content'.
        stopwords (set): Conjunto de palabras vacías a excluir del índice.

    Retorna:
        dict: Índice invertido con las frecuencias de términos (TF) e IDF.
              Estructura: {termino: {doc_id: tf, ..., '_idf': valor}, 'documentos': {...}}
    """
    indice = defaultdict(dict)
    N = len(documentos)
    indice['documentos'] = {doc['id']: doc['title'] for doc in documentos}

    for doc in tqdm(documentos, desc="Generando índice"):
        doc_id = doc["id"]
        palabras = tokenizar_texto(doc["content"], stopwords)
        tf_doc = {}
        for palabra in palabras:
            tf_doc[palabra] = tf_doc.get(palabra, 0) + 1
        for palabra, tf in tf_doc.items():
            if palabra not in indice:
                indice[palabra] = {}
            indice[palabra][doc_id] = tf

    for palabra in tqdm([t for t in indice.keys() if t != 'documentos'], desc="Calculando IDF"):
        df = len(indice[palabra])
        indice[palabra]['_idf'] = math.log(N / (1 + df))
    return indice

def guardar_indice(indice, ruta=INDICE_PATH):
    """
    Guarda el índice invertido generado en un archivo binario (.pkl).

    Parámetros:
        indice (dict): Índice invertido a guardar.
        ruta (str): Ruta donde se almacenará el archivo pickle.

    Retorna:
        None
    """
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(indice, f)
    print(f"Índice guardado en {ruta}")

def cargar_indice(ruta=INDICE_PATH):
    """
    Carga un índice invertido previamente guardado en formato pickle.

    Parámetros:
        ruta (str): Ruta del archivo .pkl que contiene el índice guardado.

    Retorna:
        dict | None: Índice cargado si existe, o None si no se encuentra el archivo.
    """
    if os.path.exists(ruta):
        with open(ruta, "rb") as f:
            return pickle.load(f)
    else:
        print("No se encontró índice guardado.")
        return None
