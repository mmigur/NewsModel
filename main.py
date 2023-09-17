from Model.prepare_data import TextPreprocess
from fastapi import FastAPI

app = FastAPI(
    title="AI news model",
    version="0.0.1"
)

@app.get('/clf_news_target/{news_text}')
def getNewsTarget(news_text: str):
    text_prepare = TextPreprocess(news_text).clear_all()
    return {"status": 200, "clear_news_text": " ".join(text_prepare)}

@app.get('/duplicate_news/{news_text_1}/{news_text_2}')
def getDuplicate(news_text_1: str, news_text_2: str):
    return 0