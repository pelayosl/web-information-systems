import re
import os
from simhash import Simhash, SimhashIndex


def get_words(s):
    """
    Convierte una cadena en minúsculas y la divide en palabras.
    Reemplaza cualquier carácter que no sea una letra, número o guión bajo por espacios.
    
    Args:
        s (str): Cadena de entrada.
    
    Returns:
        list: Lista de palabras extraídas de la cadena.
    """
    s = s.lower()
    s = re.sub(r'[^\w]+', ' ', s)
    return s.split()


def organize_sentences_mccornick(train_number):
    """
    Organiza las frases del dataset McCornick usando un patrón específico.
    
    Args:
        train_number (int): Número del conjunto de entrenamiento.
    
    Returns:
        dict: Diccionario con identificadores y frases.
    """
    pattern = re.compile(r'^(\w+)\s+(.*)$')
    if train_number != 0:
        file_name = f"simhash/train/articles_{train_number}.train"
    return organize_sentences(file_name, pattern)

def organize_sentences_quora():
    """
    Organiza las preguntas del dataset Quora usando un patrón específico.
    
    Returns:
        dict: Diccionario con identificadores y preguntas.
    """
    pattern = re.compile(r'^(\d+)\s+(.*)$')
    file_name = "simhash/quora/questions.txt"
    return organize_sentences(file_name, pattern)

def organize_sentences(file_name, pattern):
    """
    Lee un archivo y organiza sus líneas en un diccionario usando un patrón regex.
    
    Args:
        file_name (str): Ruta del archivo a leer.
        pattern (re.Pattern): Patrón regex para extraer identificador y frase.
    
    Returns:
        dict: Diccionario con identificadores y frases.
    """
    sentences = {}
    with open(file_name) as file:
        lines = file.readlines()
    for line in lines:
        match = pattern.match(line.strip())
        if match:
                identifier, sentence = match.groups()
                sentences[identifier] = sentence
    return sentences

def create_obj_words(sentences):
    """
    Crea una lista de tuplas (identificador, Simhash) para cada frase.
    
    Args:
        sentences (dict): Diccionario de identificadores y frases.
    
    Returns:
        list: Lista de tuplas (identificador, Simhash).
    """
    objs_words = []

    for k, v in sentences.items():
        sentence_id = k

        features = get_words(v)

        simhash_value = Simhash(features, f=f)

        objs_words.append((sentence_id, simhash_value))
    return objs_words


def find_near_duplicates(index, simhash_obj):
    """
    Busca duplicados cercanos en el índice Simhash.
    
    Args:
        index (SimhashIndex): Índice Simhash.
        simhash_obj (Simhash): Objeto Simhash de la frase.
    
    Returns:
        list: Lista de identificadores de duplicados cercanos.
    """
    return index.get_near_dups(simhash_obj)

def compare(sentences, k):
    """
    Compara frases para encontrar duplicados cercanos y guarda los pares únicos en un archivo.
    
    Args:
        sentences (dict): Diccionario de identificadores y frases.
        k (int): Tolerancia de SimhashIndex.
    
    Returns:
        set: Conjunto de pares únicos escritos en el archivo.
    """
    identifiers = {}

    for key in sentences.keys():
        sentence_words = Simhash(get_words(sentences.get(key)), f=f)
        near_duplicates = find_near_duplicates(index_words, sentence_words)
            
        near_duplicates = find_near_duplicates(index_words, sentence_words)
        if len(near_duplicates)>0:
            identifiers[key] = []
            for near_duplicate in near_duplicates:
                if(near_duplicate != key):
                        identifiers[key].append(near_duplicate)

    with open(f"simhash/results/results_{k}.txt", "a") as res:
        written_pairs = set()
        for key in identifiers:
            for dup in identifiers[key]:
                pair = tuple(sorted([key, dup]))
                if pair not in written_pairs:
                    res.write(f"{pair[0]} {pair[1]}\n")
                    written_pairs.add(pair)
    return written_pairs

def parse_truths(file_name):
    """
    Parsea el archivo de verdades para obtener pares de identificadores duplicados.
    
    Args:
        file_name (str): Ruta del archivo de verdades.
    
    Returns:
        set: Conjunto de pares de identificadores (ordenados).
    """
    truth_pairs = set()
    with open(file_name, "r") as t:
        for line in t.readlines():
            t1 = line.split()[0].strip()
            t2 = line.split()[1].strip()
            pair = tuple(sorted([t1, t2]))
            truth_pairs.add(pair)
        
    return truth_pairs

def compute_values(results, truths):
    """
    Calcula VP (verdaderos positivos), FP (falsos positivos) y FN (falsos negativos).
    
    Args:
        results (set): Pares encontrados por Simhash.
        truths (set): Pares verdaderos del archivo de verdades.
    
    Returns:
        tuple: VP, FP, FN.
    """
    checked_truths = set()
    VP = 0
    FP = 0
    FN = 0
    for tup in results:
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
        tuple: Precisión, exhaustividad.
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

# ------------------------- Zona de experimentación ------------------------- #

train_number = 1000
f = 128 # 64 solamente para articles_100
quora = False

if quora:
    sentences = organize_sentences_quora()
else:
    sentences = organize_sentences_mccornick(train_number)

words = create_obj_words(sentences)

# En el documento asociado se especifican los valores adecuados de 
# k para las distintas colecciones (valores que no generan problemas)
#
# Por defecto, se usan [0, 3, 5, 10]
tolerances = [0, 3, 5, 10]


for i in range(len(tolerances)):
    if os.path.exists(f'simhash/results/results_{tolerances[i]}.txt'):
            os.remove(f'simhash/results/results_{tolerances[i]}.txt')
            
    index_words = SimhashIndex(words, k=tolerances[i], f=f)

    written_pairs = compare(sentences, tolerances[i])
    if quora:
        truths_file = "simhash/quora/duplicates.txt"
    else:
        truths_file = f"simhash/truth/articles_{train_number}.truth"
        
    VP, FP, FN = compute_values(written_pairs, parse_truths(truths_file))
    precision, recall = compute_precision_recall(VP, FP, FN)
    F1 = compute_F1(precision, recall)

    print(f"Tolerance     = {tolerances[i]}")
    print(f"VP            = {VP}")
    print(f"FP            = {FP}")
    print(f"FN            = {FN}")
    print(f"Precisión     = {precision}")
    print(f"Exhaustividad = {recall}")
    print(f"F1 = {F1}")
    print("------------------------------------------")