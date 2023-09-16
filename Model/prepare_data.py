import string
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk import ngrams

from pymorphy2 import MorphAnalyzer

N = 2
text = (
'''
👧👦 В загородных детских лагерях Марий Эл в данный момент отдыхают 2335 детей, и ещё 12265 ребят посещают 175 пришкольных лагерей.

Всего на территории республики планируется открыть 200 оздоровительных организаций для детей и подростков, из которых в первую смену будут функционировать 192 лагеря, включая 13 загородных, 177 пришкольных и 2 палаточных.

Все организации, которые начнут работу в первую смену, имеют санитарно-эпидемиологические заключения от Роспотребнадзора. В некоторых лагерях, расположенных на берегу озер, дети смогут купаться, ведь их администрация получила разрешение от надзорной инстанции.
'''
)

nltk.download('punkt')
STOPWORDS_AND_CHARS = stopwords.words('russian')
emoji_finder = re.compile('[\U0001F300-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BF]+')

COUNT_RUSSIAN_LETTERS = 33
RUSSIAN_ALPHABET = [chr(0x0410 + index) for index in range(COUNT_RUSSIAN_LETTERS)]
RUSSIAN_ALPHABET.extend([chr(0x0430 + index) for index in range(COUNT_RUSSIAN_LETTERS)])
STOPWORDS_AND_CHARS.extend(string.punctuation)
STOPWORDS_AND_CHARS.extend(RUSSIAN_ALPHABET)

def remove_digits(text: str) -> str:
    return [word for word in text if (word not in string.digits) and (word not in string.punctuation)]

def remove_stop_words(text: str, stopwords=STOPWORDS_AND_CHARS) -> list:
    return [word for word in text.split(' ') if word not in stopwords]

def lemmatize(text) -> str:
    pymorphy2_analyzer = MorphAnalyzer()
    return ' '.join([pymorphy2_analyzer.parse(word)[0].normal_form.strip() for word in text.split(' ')]).strip()

def tokenize(text) -> list:
    tokenized = sent_tokenize(text)
    tokens = []
    for token in tokenized:
        with_no_tabs = token.replace('\n', '').replace('\xa0', '')
        with_no_emoji = re.sub(emoji_finder, '', with_no_tabs)
        tokens.append(with_no_emoji.strip())

    return tokens


def to_bigrams(tokens: list):
    return ngrams(''.join(tokens).split(), N)


after_remove_dg = remove_digits(text)
print(f'после цифр: {after_remove_dg}')
after_remove_stop_words = remove_stop_words(''.join(after_remove_dg))
print(f'после стоп слов: {after_remove_stop_words}')
after_lemmatize = lemmatize(' '.join(after_remove_stop_words))
print(f'после морфы: {after_lemmatize}')
after_tokenize = tokenize(after_lemmatize)
print(f'токены: {after_tokenize}')
print(*to_bigrams(after_tokenize))