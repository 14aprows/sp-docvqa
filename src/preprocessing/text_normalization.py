import re 
import unicodedata

def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)

def compact_normalize(text):
    text = normalize_text(text)
    return "".join(character for character in text if character.isalnum())