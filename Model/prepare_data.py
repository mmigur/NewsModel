import string
import re
from dataclasses import dataclass

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk import ngrams

from pymorphy2 import MorphAnalyzer


nltk.download('stopwords')
nltk.download('punkt')
STOPWORDS_AND_CHARS = stopwords.words('russian')
emoji_finder = re.compile('[\U0001F300-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BF]+')

COUNT_RUSSIAN_LETTERS = 33
RUSSIAN_ALPHABET = [chr(0x0410 + index) for index in range(COUNT_RUSSIAN_LETTERS)]
RUSSIAN_ALPHABET.extend([chr(0x0430 + index) for index in range(COUNT_RUSSIAN_LETTERS)])
STOPWORDS_AND_CHARS.extend(string.punctuation)
STOPWORDS_AND_CHARS.extend(RUSSIAN_ALPHABET)

#def to_bigrams(tokens: list):
#    return ngrams(''.join(tokens).split(), N)


@dataclass
class TextPreprocess:
    ''' все методы для подготовки текста перед векторизацией '''
    stopwords = STOPWORDS_AND_CHARS
    post: str # может быть и спискос, это просто значение,
              # которое хранит в себе все промежуточные результаты и финальный

    def clear_all(self):
        # удаление ненужных символов
        self.remove_shit()
        # приведеие в нормальную форму
        self.lemmatize()
        # токенизация
        self.tokenize()

    def remove_shit(self) -> None:
        # удаление ссылок и символов переноса строки и тире
        without_emojis_n_tabs = re.sub(emoji_finder, '', self.post).replace('\n', '').replace('\xa0', '')
        without_links = re.sub(r'^https?:\/\/.*[\r\n]*', '', without_emojis_n_tabs, flags=re.MULTILINE)
        # удаление цифр и стоп слов
        without_digits = (
            ''.join([word for word in without_links if (word not in string.punctuation) and (word not in string.digits)])
        )
        self.post = ' '.join([word for word in without_digits.split(' ') if word not in self.stopwords])

    def lemmatize(self) -> None:
        pymorphy2_analyzer = MorphAnalyzer()
        self.post = ' '.join([pymorphy2_analyzer.parse(word)[0].normal_form.strip() for word in self.post.split(' ')]).strip()

    def tokenize(self) -> None:
        self.post = sent_tokenize(' '.join([word.strip() for word in self.post.split(' ')]))


if __name__ == '__main__':
    # только для теста, обычно это улсовие не выполняется

    processor = TextPreprocess(post='''
    👧👦 В загородных детских лагерях Марий Эл в данный момент отдыхают 2335 детей, и ещё 12265 ребят посещают 175 пришкольных лагерей.

    Всего на территории республики планируется открыть 200 оздоровительных организаций для детей и подростков, из которых в первую смену будут функционировать 192 лагеря, включая 13 загородных, 177 пришкольных и 2 палаточных.

    Все организации, которые начнут работу в первую смену, имеют санитарно-эпидемиологические заключения от Роспотребнадзора. В некоторых лагерях, расположенных на берегу озер, дети смогут купаться, ведь их администрация получила разрешение от надзорной инстанции.
    ''')
    processor.clear_all()
    print(processor.post)
