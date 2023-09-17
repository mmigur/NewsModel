import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from prepare_data import TextPreprocess

class DeleteDuplicte:

    def __init__(self, text_df):
        self.df = text_df.sample(n=500, random_state=42)

    def remove_duplicate(self):
        self.df['result_text'] = self.df['text']
        self.df['result_text'] = self.df['result_text'].apply(str)
        self.df['result_text'] = self.df['result_text'].apply(TextPreprocess.remove_shit)
        self.df['result_text'] = self.df['result_text'].apply(TextPreprocess.lemmatize)

        self.tf_idf_matrix = TextPreprocess.vectorize(self.df['result_text'])
            
        self.similarity_matrix = cosine_similarity(self.tf_idf_matrix)
        n = self.similarity_matrix.shape[0]

        indices_to_keep = list(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if self.similarity_matrix[i, j] > 0.6:
                    if j in indices_to_keep:
                        self.df = self.df.drop(
                            index=self.df.sample(n=500, random_state=42).index.values[j],
                            axis=0
                        )

        return self.df