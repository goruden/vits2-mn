
import re
from unidecode import unidecode
from phonemizer import phonemize
from text.mongolian import mongolian_to_ipa

_whitespace_re = re.compile(r'\s+')

def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)

def convert_to_ascii(text):
    return unidecode(text)

def mongolian_cleaners(text):
    text = convert_to_ascii(text)
    text = text.lower()
    text = collapse_whitespace(text)
    text = mongolian_to_ipa(text)
    return text

