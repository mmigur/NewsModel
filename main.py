import pandas as pd
from Model.delete_duplicate_model import DeleteDuplicte
from fastapi import FastAPI


app = FastAPI(
    title="AI news model",
    version="0.0.1"
)

@app.get('/DataSetProcessing/{data_frame}')
def getNewsTarget(data_frame: str):
    post_df = pd.read_excel('./Notebooks/Data/posts (1).xlsx')
    dd = DeleteDuplicte(post_df).remove_duplicate()
    return {"status": 200, "removed_duplicate_dataframe": f"{dd.values}"}