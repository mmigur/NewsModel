from Model.delete_duplicate_model import DeleteDuplicte
from Model.api_utils import get_categorired_dataframe, pd # pandas
from Model.real_config import CANDIDATE_LABELS

from fastapi import FastAPI


app = FastAPI(
    title="AI news model",
    version="1.0.0"
)

@app.get('/get-filtered')
def getFilteredDataset():
    posts = pd.read_csv('./base_dataset.csv')
    categorized = get_categorired_dataframe(posts=posts, candidate_labels=CANDIDATE_LABELS)

    without_duplicates = [
        DeleteDuplicte(dataframe).remove_duplicate() for dataframe in categorized.values()
    ]

    pd.concat(without_duplicates, ignore_index=False).to_csv('final_dataset.csv', index=False)

    return {"status": 200, "clear_news_text": "ok"}


""" # разбить каждую категорию на pandas dataframe
@app.get('/clf_news_target/{news_text}')
def getNewsTarget(news_text: str):
    text_prepare = TextPreprocess(news_text).clear_all()
    return {"status": 200, "clear_news_text": "ok"}


data_frame = pd.read_excel('./Notebooks/Data/posts (1).xlsx')
print(f"Набор данных до: {data_frame.shape}")

dd = DeleteDuplicte(data_frame).remove_duplicate()
print(f"Набор данных после: {dd.keys()}") """