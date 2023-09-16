import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax


# Эта модель полное говно если че пока что
# Загрузка предварительно обученной модели BERT и токенизатора
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=9)

# Функция для классификации текста
def classify_text(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = softmax(logits, dim=1)

    category_names = [
        "Финансы", "Технологии", "Политика", "Шоу-биз", "Fashion", "Крипта",
        "Путешествия/релокация", "Образовательный контент", "Развлечения", "Общее"
    ]
    
    predicted_labels = torch.argmax(probabilities, dim=1)
    predicted_categories = [category_names[label] for label in predicted_labels]
    
    return predicted_categories, probabilities

# Список текстов для классификации
texts = [
    "Это текст о финансах и экономике.",
    "Новости о последних технологических достижениях.",
    "Политические новости и анализ событий.",
    "Скандалы и события из мира шоу-бизнеса.",
    "Мода и стиль в одежде.",
    "Криптовалюты и блокчейн технологии.",
    "Путешествия и советы по релокации.",
    "Образовательный контент и обучение.",
    "Развлекательные новости и события.",
    "Общие новости и статьи."
]

predicted_categories, probabilities = classify_text(texts)

for text, category, prob in zip(texts, predicted_categories, probabilities):
    print(f"Текст: {text}")
    print(f"Категория: {category}")
    print(f"Вероятности: {prob}")
    print()
