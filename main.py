"""from Model.prepare_data import TextPreprocess
from fastapi import FastAPI

app = FastAPI(
    title="AI news model",
    version="0.0.1"
)

@app.get('/clf_news_target/{news_text}')
def getNewsTarget(news_text: str):
    text_prepare = TextPreprocess(news_text).clear_all()
    return {"status": 200, "clear_news_text": "ok"}"""



import pandas as pd
from Model.model import DeleteDuplicte


data_frame = pd.read_excel('./Notebooks/Data/posts (1).xlsx')
print(f"Набор данных до: {data_frame.shape}")

dd = DeleteDuplicte(data_frame).remove_duplicate()
print(f"Набор данных после: {dd.shape}")