import pandas as pd
from transformers import pipeline


classifier = pipeline("zero-shot-classification")

posts = pd.read_excel('./Model/Notebooks/Data/ai_bot_app_post.xls')
candidate_labels = "Финансы, Технологии, Политика, Шоубиз, Fashion, Крипта, Путешествия/релокация, Образовательный контент, Развлечения, Общее".split(', ')


output = classifier(list(posts['title'][:5]), candidate_labels)
print(output)

'''
1) как работать с русским языком
2) как рабоать с GPU
3) попробовать скачать несколько других моделей, 
которые могут классифицировать текст без таргета
'''
