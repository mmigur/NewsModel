import string
import re
from dataclasses import dataclass

import torch
import nltk
from nltk.corpus import stopwords

from pymorphy2 import MorphAnalyzer

nltk.download('stopwords')
nltk.download('punkt')

STOPWORDS_AND_CHARS = stopwords.words('russian') # скачиваем стоп слова.

COUNT_RUSSIAN_LETTERS = 33 # кол-во букв в русском алфивите.

# получаем все буквы русского языка в нижнем и верхнем регистре.
RUSSIAN_ALPHABET = [chr(0x0410 + index) for index in range(COUNT_RUSSIAN_LETTERS)] 
RUSSIAN_ALPHABET.extend([chr(0x0430 + index) for index in range(COUNT_RUSSIAN_LETTERS)])

# Дополняем список стоп слов знаками пунктуации и буквами русского алфавита.
STOPWORDS_AND_CHARS.extend(string.punctuation)
STOPWORDS_AND_CHARS.extend(RUSSIAN_ALPHABET)

# regex для поиска эмоджи.
emoji_finder = re.compile('[\U0001F300-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BF]+')

@dataclass
class TextPreprocess:
    """ Класс для обработки текста перед, входом в модель. """
    def remove_shit(text) -> str:
        """
        Метод удаления ненужных символов, а так же эмоджи. Метод оставляет ссылки при помощи regex.
        text: текст над которым будет проведена обработка.
        """

        # удаление из текста эмоджи и оставляем ссылки.
        without_emojis_n_tabs = re.sub(emoji_finder, '', text).replace('\n', '').replace('\xa0', '')
        without_links = re.sub(r'^https?:\/\/.*[\r\n]*', '', without_emojis_n_tabs, flags=re.MULTILINE)

        # удаляем знаки пунктуации и цифры
        without_digits = (
            ''.join([word for word in without_links if (word not in string.punctuation) and (word not in string.digits)])
        )
        return ' '.join([word for word in without_digits.split(' ') if word not in STOPWORDS_AND_CHARS])

    def lemmatize(text) -> str:
        """
        Метод позволяющий привести слова к нормальной форме.
        text: текст над которым будет проведена обработка.
        """

        # приводим слова в нормальную форму при помощи PyMorphy.
        pymorphy2_analyzer = MorphAnalyzer()
        return ' '.join([pymorphy2_analyzer.parse(word)[0].normal_form.strip() for word in text.split(' ')]).strip()

    def embed_bert_cls(text, model, tokenizer):
        """
        Пред обученный токенизатор, а так же модель которая генерирует эмбеддинги.
        text: текст над которым будет проведена обработка.
        model: предобученная модель для генерации эмбеддингов.
        """
        t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**{k: v.to(model.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, :, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        
        return embeddings.cpu().mean(dim = 1).squeeze(0).numpy()