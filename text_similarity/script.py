from textblob import TextBlob, Word
from nltk.corpus import stopwords
import re
import math
import os

stops = set(stopwords.words('english'))
def main(boolean_search):
    """
    Función principal que procesa las consultas y textos, calculando la similitud entre ellos
    usando diferentes métricas (dice, jaccard, cosine, overlapping).
    
    Args:
        boolean_search (bool): True para búsqueda booleana, False para búsqueda por frecuencias
    """
    if os.path.exists(f'lab4/results.txt'):
            os.remove(f'lab4/results.txt')
    texts = load_lines("lab4/cran-1400.txt")
    queries = load_lines("lab4/cran-queries.txt")
    counter = 0
    with open("lab4/results.txt", "a") as file:
        file.write("SEARCH TYPE: BOOLEAN\n") if boolean_search else file.write("SEARCH TYPE: FREQUENCIES\n")
        for q_id, q_text in queries.items():
            file.write(f"QUERY: {q_id}\n")
            counter += 1

            best_text, best_score = find_best_text(q_text, texts, "dice", boolean_search)
            res = f"DICE: {best_text} ({best_score} score)\n"
            file.write(res)

            best_text, best_score = find_best_text(q_text, texts, "jaccard", boolean_search)
            res = f"JACCARD: {best_text} ({best_score} score)\n"
            file.write(res)

            best_text, best_score = find_best_text(q_text, texts, "cosine", boolean_search)
            res = f"COSINE: {best_text} ({best_score} score)\n"
            file.write(res)

            best_text, best_score = find_best_text(q_text, texts, "overlapping", boolean_search)
            res = f"OVERLAPPING: {best_text} ({best_score} score)\n-------------------------\n"
            file.write(res)
        

def load_lines(file):
    """
    Carga y procesa un archivo de texto, creando un diccionario donde las claves son los
    identificadores y los valores son bolsas de palabras de los textos.
    
    Args:
        file (str): Ruta del archivo a procesar
    
    Returns:
        dict: Diccionario con los identificadores y sus bolsas de palabras correspondientes
    """
    dictionary = {}
    with open(file, "r") as cran:
        lines = cran.readlines()
    
    pattern = r'^(I\s*\d+)\s+(.*)$'
    for line in lines:
        match = re.match(pattern, line)
        if match:
            identifier, text = match.groups()
            dictionary[identifier] = string_to_bag_of_words(text)
    return dictionary

def string_to_bag_of_words(text):
    """
    Convierte un texto en una bolsa de palabras (diccionario de frecuencias).
    Realiza tokenización, conversión a minúsculas, eliminación de stopwords y lematización.
    
    Args:
        text (str): Texto a procesar
    
    Returns:
        dict: Diccionario con las palabras como claves y sus frecuencias como valores
    """
    bow = {}
    zen = TextBlob(text)
    tokens = zen.words # tokenizar
    for token in tokens:
        token = token.lower() # minúsculas
        if token in stops: continue # ignorar si es una stopword
        token = Word(token).lemmatize() # lema
        bow[token] = bow.get(token, 0) + 1 # se almacena la frecuencia de aparición
    return bow

def find_best_text(query, texts, metric, boolean_search):
    """
    Encuentra el texto más similar a una consulta usando la métrica especificada.
    
    Args:
        query (dict): Consulta a comparar, convertida a bolsa de palabras
        texts (dict): Diccionario con los textos a comparar, cada texto convertido a bolsa de palabras')
        boolean_search (bool): True para búsqueda booleana, False para búsqueda por frecuencias
    
    Returns:
        tuple: (identificador del mejor documento, puntuación más alta)
    """
    score = 0
    best_score = 0
    best_doc = None
    for doc_id, content in texts.items():
        doc_bow = texts.get(doc_id) 
        if metric == "dice":
            score = dice_coefficient(query, doc_bow, boolean_search)
        elif metric == "jaccard":
            score = jaccard_coefficient(query, doc_bow, boolean_search)
        elif metric == "cosine":
            score = cosine(query, doc_bow, boolean_search)
        elif metric == "overlapping":
            score = overlapping_coefficient(query, doc_bow, boolean_search)
        else:
            score = 0
        if score > best_score:
            best_score = score
            best_doc = doc_id
    return best_doc, best_score

def dice_coefficient(q_bow, doc_bow, boolean_search):
    """
    Calcula el coeficiente de Dice entre dos bolsas de palabras.
    
    Args:
        q_bow (dict): Bolsa de palabras de la consulta
        doc_bow (dict): Bolsa de palabras del documento
        boolean_search (bool): True para cálculo booleano, False para cálculo por frecuencias
    
    Returns:
        float: Coeficiente de Dice entre 0 y 1
    """
    # aplicación booleana
    if boolean_search:
        set1, set2 = set(q_bow.keys()), set(doc_bow.keys())
        intersection = len(set1 & set2)
        return (2 * intersection) / (len(set1) + len(set2))

    # aplicación por frecuencias
    intersection_sum = 0
    sum_q = sum(q_bow.values())
    sum_d = sum(doc_bow.values())
    for term in q_bow:
        if term in doc_bow:
            intersection_sum += min(q_bow[term], doc_bow[term])

    denominator = sum_q + sum_d
    return (2 * intersection_sum) / denominator if denominator > 0 else 0.0


def jaccard_coefficient(q_bow, doc_bow, boolean_search):
    """
    Calcula el coeficiente de Jaccard entre dos bolsas de palabras.
    
    Args:
        q_bow (dict): Bolsa de palabras de la consulta
        doc_bow (dict): Bolsa de palabras del documento
        boolean_search (bool): True para cálculo booleano, False para cálculo por frecuencias
    
    Returns:
        float: Coeficiente de Jaccard entre 0 y 1
    """
    # aplicación booleana
    if boolean_search:
        set1, set2 = set(q_bow.keys()), set(doc_bow.keys())
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return (intersection/union)
    
    # aplicación por frecuencias
    intersection_sum = 0
    union_sum = 0
    all_terms = set(q_bow.keys()) | set(doc_bow.keys()) 
    for term in all_terms:
        f1 = q_bow.get(term, 0)
        f2 = doc_bow.get(term, 0)
        intersection_sum += min(f1, f2)
        union_sum += max(f1, f2)

    return (intersection_sum/union_sum) if union_sum > 0 else 0.0

def cosine(q_bow, doc_bow, boolean_search):
    """
    Calcula la similitud del coseno entre dos bolsas de palabras.
    
    Args:
        q_bow (dict): Bolsa de palabras de la consulta
        doc_bow (dict): Bolsa de palabras del documento
        boolean_search (bool): True para cálculo booleano, False para cálculo por frecuencias
    
    Returns:
        float: Similitud del coseno entre 0 y 1
    """
    # aplicación booleana
    if boolean_search:
        set1, set2 = set(q_bow.keys()), set(doc_bow.keys())
        intersection = len(set1 & set2)
        mult = math.sqrt(len(set1) * len(set2))
        return (intersection/mult)
    
    # aplicación por frecuencias
    numerator = 0
    sum_q_sq = 0
    sum_d_sq = 0

    all_terms = set(q_bow.keys()) | set(doc_bow.keys())

    for term in all_terms:
        f1 = q_bow.get(term, 0)
        f2 = doc_bow.get(term, 0)
        numerator += f1*f2
        sum_q_sq += f1**2
        sum_d_sq += f2**2

    denominator = math.sqrt(sum_q_sq) * math.sqrt(sum_d_sq)
    return numerator / denominator if denominator > 0 else 0.0

def overlapping_coefficient(q_bow, doc_bow, boolean_search):
    """
    Calcula el coeficiente de solapamiento entre dos bolsas de palabras.
    
    Args:
        q_bow (dict): Bolsa de palabras de la consulta
        doc_bow (dict): Bolsa de palabras del documento
        boolean_search (bool): True para cálculo booleano, False para cálculo por frecuencias
    
    Returns:
        float: Coeficiente de solapamiento entre 0 y 1
    """
    # aplicación booleana
    if boolean_search:
        set1, set2 = set(q_bow.keys()), set(doc_bow.keys())
        intersection = len(set1 & set2)
        denominator = min(len(set1), len(set2))
        return intersection / denominator
    
    # aplicación por frecuencias
    intersection_sum = 0
    all_terms = set(q_bow.keys()) | set(doc_bow.keys()) 
    sum_q = sum(q_bow.values())
    sum_d = sum(doc_bow.values())
    for term in all_terms:
        f1 = q_bow.get(term, 0)
        f2 = doc_bow.get(term, 0)
        intersection_sum += min(f1, f2)
    
    denominator = min(sum_q, sum_d)
    return intersection_sum / denominator if denominator > 0 else 0.0

# ------------------ Ejecución ------------------ #
boolean_search = False
main(boolean_search)
