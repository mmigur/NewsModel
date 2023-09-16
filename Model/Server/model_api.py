from fastapi import FastAPI

app = FastAPI(
    title="AI news model",
    version="0.0.1"
)

@app.get('/clf_news_theme/{news_text}')
def getData(news_text: str):
    return {'status': 200, 'data': 'Новость классифицирована'}

@app.get('/duplicate_news/{news_text}')
def getDuplicate(news_text: str):
    return {'status': 200, 'data': 'Дубликаты новостей удалены!'}