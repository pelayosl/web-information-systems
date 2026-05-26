import math
from collections import Counter
from tqdm import tqdm
import bm25s
from .utils import tokenizar_texto

def buscar_documentos(consulta, indice, documentos, operador="AND", stopwords=None):
    """
    Busca documentos relevantes para una consulta según un operador booleano (AND / OR).

    Parámetros:
        consulta (str): Texto de la consulta a procesar.
        indice (dict): Índice invertido donde las claves son términos y los valores son diccionarios
                       con los documentos en los que aparecen y su frecuencia.
        documentos (list): Lista de documentos del corpus, cada uno con su 'id' y 'content'.
        operador (str): Tipo de operador booleano ("AND" o "OR") usado para combinar términos.
        stopwords (set): Conjunto de palabras vacías que se filtrarán en la tokenización.

    Retorna:
        dict: Diccionario con los IDs de documentos como claves y valores 1 (relevante) o 0 (no relevante).
    """
    terminos = tokenizar_texto(consulta, stopwords)
    ids_documentos = [doc["id"] for doc in documentos]
    relevancia = {doc_id: 0 for doc_id in ids_documentos}
    terminos_existentes = [t for t in terminos if t in indice and t != "documentos"]
    if not terminos_existentes:
        return relevancia
    if operador.upper() == "AND":
        docs_relevantes = set(indice[terminos_existentes[0]].keys()) - {"_idf"}
        for t in terminos_existentes[1:]:
            docs_relevantes &= set(indice[t].keys()) - {"_idf"}
    else:
        docs_relevantes = set()
        for t in terminos_existentes:
            docs_relevantes |= set(indice[t].keys()) - {"_idf"}
    for doc_id in docs_relevantes:
        relevancia[doc_id] = 1
    return relevancia

def similitud_coseno(consulta, indice, documentos, stopwords):
    """
    Calcula la similitud del coseno entre una consulta y los documentos relevantes del corpus.

    Parámetros:
        consulta (str): Texto de la consulta.
        indice (dict): Índice invertido con términos, frecuencias y valores IDF.
        documentos (list): Lista de documentos (diccionarios con 'id' y 'content').
        stopwords (set): Palabras vacías que deben eliminarse antes del cálculo.

    Retorna:
        dict: Diccionario con los IDs de documentos y su similitud coseno (float).
    """
    terminos = tokenizar_texto(consulta, stopwords)
    relevancia_or = buscar_documentos(consulta, indice, documentos, "OR", stopwords)
    docs_relevantes = [doc_id for doc_id, val in relevancia_or.items() if val == 1]
    if not docs_relevantes:
        return {}

    tf_consulta = Counter(terminos)
    vector_consulta = {}
    for termino, tf in tf_consulta.items():
        if termino in indice and "_idf" in indice[termino]:
            vector_consulta[termino] = tf * indice[termino]["_idf"]

    similitudes = {}
    for doc_id in tqdm(docs_relevantes, desc="Calculando similitud coseno"):
        vector_doc = {}
        for termino, datos in indice.items():
            if termino in ("documentos",):
                continue
            if doc_id in datos:
                tf = datos[doc_id]
                idf = datos.get("_idf", 0)
                vector_doc[termino] = tf * idf
        num = sum(vector_doc.get(t, 0) * vector_consulta.get(t, 0) for t in vector_consulta)
        denom = math.sqrt(sum(v**2 for v in vector_doc.values())) * math.sqrt(sum(v**2 for v in vector_consulta.values()))
        similitudes[doc_id] = num / denom if denom else 0
    return similitudes

def ranking_bm25_lisa(consulta, documentos, flavour="robertson", k=None, stopwords=None):
    """
    Genera un ranking de documentos usando el modelo BM25 con distintos sabores (flavours).

    Parámetros:
        consulta (str): Texto de la consulta.
        documentos (list): Lista de documentos del corpus, cada uno con 'id' y 'content'.
        flavour (str): Variante del algoritmo BM25 (robertson, lucene, atire, bm25l, bm25+).
        k (int): Número de documentos a recuperar. Si no se especifica, devuelve todos.
        stopwords (set): Conjunto de palabras vacías a eliminar en la tokenización.

    Retorna:
        dict: Diccionario {doc_id: puntuación BM25}.
    """
    flavour = flavour.lower()
    valid_flavours = ['robertson', 'lucene', 'atire', 'bm25l', 'bm25+']
    if flavour not in valid_flavours:
        raise ValueError(f"Flavour no válido. Usa uno de: {', '.join(valid_flavours)}")
    tokenized_corpus = [tokenizar_texto(doc["content"], stopwords) for doc in documentos]
    retriever = bm25s.BM25(method=flavour)
    retriever.index(tokenized_corpus)
    query_tokens = tokenizar_texto(consulta, stopwords)
    if k is None:
        k = len(documentos)
    docs_arr, scores_arr = retriever.retrieve([query_tokens], k=k, return_as="tuple")
    resultados = {documentos[idx]["id"]: float(score)
                  for idx, score in zip(docs_arr[0], scores_arr[0])}
    return resultados
