import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd

# Загружаем модель BERT для классификации
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Список категорий
posts = pd.read_excel(r'C:\Users\anama\Downloads\posts (1).xlsx')
categories = ['Финансы', 'Технологии', 'Политика', 'Шоубиз', 'Fashion', 'Крипта', 'Путешествия/релокация', 'Образовательный контент', 'Развлечения', 'Общеe']

def classify_text(text):
    # Токенизируем текст и добавляем специальные токены [CLS] и [SEP]
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    # Подаем векторизованный текст на вход модели BERT
    outputs = model(**inputs)

    # Получаем вероятности для каждой категории
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]

    # Определяем наиболее вероятную категорию
    max_prob_index = probabilities.index(max(probabilities))
    predicted_category = categories[max_prob_index]

    return predicted_category

for post in posts['text'][:30]:
    print(f'текст: {post}')
    print(f'категория: {classify_text(post)}')