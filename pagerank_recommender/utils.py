import re

def string_to_bag_of_words(text):
    words = text.strip().split("|")
    return words

def jaccard(words1, words2):
    set1, set2 = set(words1), set(words2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return (intersection/union)

def safe_filename(title):
    """Limpia el título de caracteres no válidos para nombres de archivo."""
    return re.sub(r'[\\/*?:"<>|]', "", title.replace(" ", "_"))
