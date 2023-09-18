import pandas as pd
from transformers import pipeline


classifier = pipeline(
    task="zero-shot-classification",
	model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
)

posts = pd.read_excel('./Model/Notebooks/Data/posts (1).xlsx')
candidate_labels = [
    "Блоги",
    "Новости и СМИ",
    "Развлечения и юмор",
    "Технологии",
    "Экономика",
    "Бизнес и стартапы",
    "Криптовалюты",
    "Путешествия",
    "Маркетинг, PR, реклама",
    "Психология",
    "Дизайн",
    "Политика",
    "Искусство",
    "Право",
    "Образование и познавательное",
    "Спорт",
    "Мода и красота",
    "Здоровье и медицина",
    "Картинки и фото",
    "Софт и приложения",
    "Видео и фильмы",
    "Музыка",
    "Игры",
    "Еда и кулинария",
    "Цитаты",
    "Рукоделие",
    "Финансы",
    "Шоубиз",
    "Другое",
]


output = classifier(
    list(posts['title'][24:25]), candidate_labels=candidate_labels
)
print(output)


'''
1) как работать с русским языком
2) как рабоать с GPU
3) попробовать скачать несколько других моделей, 
которые могут классифицировать текст без таргета
'''
