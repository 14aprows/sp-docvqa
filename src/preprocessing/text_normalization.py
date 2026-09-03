import re
import unicodedata

def normalize_data(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def compact_normalize(text):
    text = unicodedata.normalize("NFKC", str(text))
    return "".join(character for character in text if character.isalnum())