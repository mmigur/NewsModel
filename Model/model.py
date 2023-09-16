import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Загружаем модель BERT для классификации
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Список категорий
categories = ["Технологии", "Fashion", "Шоу-биз", "Общее"]

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

text = "Смартфоны и гаджеты на базе Android"
predicted_category = classify_text(text)
print("Predicted category:", predicted_category)