
import re

_latin_to_mongolian = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
    ('a', 'а'), ('b', 'б'), ('c', 'ц'), ('d', 'д'), ('e', 'э'), ('f', 'ф'),
    ('g', 'г'), ('h', 'х'), ('i', 'и'), ('j', 'ж'), ('k', 'к'), ('l', 'л'),
    ('m', 'м'), ('n', 'н'), ('o', 'о'), ('p', 'п'), ('q', 'к'), ('r', 'р'),
    ('s', 'с'), ('t', 'т'), ('u', 'у'), ('v', 'в'), ('w', 'в'), ('x', 'кс'),
    ('y', 'й'), ('z', 'з'),
]]

_digit_to_mongolian = {
    '0': 'тэг', '1': 'нэг', '2': 'хоёр', '3': 'гурав', '4': 'дөрөв',
    '5': 'тав', '6': 'зургаа', '7': 'долоо', '8': 'найм', '9': 'ес',
}

def latin_to_cyrillic(text):
    for regex, replacement in _latin_to_mongolian:
        text = re.sub(regex, replacement, text)
    return text

def digits_to_mongolian(text):
    for digit, word in _digit_to_mongolian.items():
        text = text.replace(digit, word)
    return text

def mongolian_to_ipa(text):
    text = latin_to_cyrillic(text)
    text = digits_to_mongolian(text)
    return text

