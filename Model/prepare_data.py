import string
import re
from dataclasses import dataclass

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from pymorphy2 import MorphAnalyzer

nltk.download('stopwords')
nltk.download('punkt')

STOPWORDS_AND_CHARS = stopwords.words('russian')

COUNT_RUSSIAN_LETTERS = 33
RUSSIAN_ALPHABET = [chr(0x0410 + index) for index in range(COUNT_RUSSIAN_LETTERS)]
RUSSIAN_ALPHABET.extend([chr(0x0430 + index) for index in range(COUNT_RUSSIAN_LETTERS)])

STOPWORDS_AND_CHARS.extend(string.punctuation)
STOPWORDS_AND_CHARS.extend(RUSSIAN_ALPHABET)
emoji_finder = re.compile('[\U0001F300-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BF]+')

@dataclass
class TextPreprocess:
    ''' все методы для подготовки текста перед векторизацией '''
    stopwords = STOPWORDS_AND_CHARS
    post: str

    def clear_all(self):
        self.remove_shit()
        self.lemmatize()
        self.tokenize()

        return self.post

    def remove_shit(self) -> None:
        without_emojis_n_tabs = re.sub(emoji_finder, '', self.post).replace('\n', '').replace('\xa0', '')
        without_links = re.sub(r'^https?:\/\/.*[\r\n]*', '', without_emojis_n_tabs, flags=re.MULTILINE)

        without_digits = (
            ''.join([word for word in without_links if (word not in string.punctuation) and (word not in string.digits)])
        )
        self.post = ' '.join([word for word in without_digits.split(' ') if word not in self.stopwords])

    def lemmatize(self) -> None:
        pymorphy2_analyzer = MorphAnalyzer()
        self.post = ' '.join([pymorphy2_analyzer.parse(word)[0].normal_form.strip() for word in self.post.split(' ')]).strip()

    def tokenize(self) -> None:
        self.post = sent_tokenize(' '.join([word.strip() for word in self.post.split(' ')]))
