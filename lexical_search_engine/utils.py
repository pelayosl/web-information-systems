import os
import re
import Stemmer

stemmer = Stemmer.Stemmer('english')

def cargar_stopwords(ruta):
    """
    Carga un conjunto de palabras vacías (stopwords) desde un archivo de texto.

    Parámetros:
        ruta (str): Ruta al archivo que contiene una stopword por línea.

    Retorna:
        set: Conjunto de palabras vacías en minúsculas.
             Si el archivo no existe, devuelve un conjunto vacío y muestra una advertencia.
    """
    if not os.path.exists(ruta):
        print(f"Advertencia: no se encontró {ruta}. No se eliminarán stopwords.")
        return set()
    with open(ruta, "r", encoding="utf-8") as f:
        stopwords = {line.strip().lower() for line in f if line.strip()}
    return stopwords

def tokenizar_texto(texto, stopwords=None):
    """
    Tokeniza un texto aplicando limpieza, normalización, eliminación de stopwords y stemming.

    Parámetros:
        texto (str): Texto de entrada a procesar.
        stopwords (set, opcional): Conjunto de palabras vacías a eliminar.

    Retorna:
        list: Lista de tokens procesados (lematizados mediante stemming).
    """
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
    tokens = texto.split()
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return stemmer.stemWords(tokens)
