from fastapi import FastAPI

app = FastAPI(
    title="AI news model",
    version="0.0.1"
)

@app.get('/clf_news_theme/{news_text}')
def getData(news_text: str):
    return {'status': 200, 'data': ' '.join(news_text)}