import numpy as np

from Model.prepare_data import TextPreprocess
from transformers import AutoTokenizer, AutoModel

class DeleteDuplicte:
    """ Класс для удаления дубликатов из набора данных. """

    def __init__(self, text_df):
        self.indexes = text_df.sample(n=100, random_state=42).index.values
        self.df = text_df.sample(n=100, random_state=42)
        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
        self.model = AutoModel.from_pretrained("cointegrated/rubert-tiny")

    def remove_duplicate(self):
        """
        Функция для удаления дубликатов.
        """
        self.df['result_text'] = self.df['text']
        self.df['result_text'] = self.df['result_text'].apply(str)
        self.df['result_text'] = self.df['result_text'].apply(TextPreprocess.remove_shit) # удаление мусора.
        self.df['result_text'] = self.df['result_text'].apply(TextPreprocess.lemmatize) # приведение слов к нормальной форме.

        # токенизация и генерация эмбеддингов.
        self.df['vectors'] = self.df['result_text'].apply(lambda x: TextPreprocess.embed_bert_cls(x,self.model, self.tokenizer))
        self.df['vec_length'] = self.df['vectors'].apply(lambda x: np.linalg.norm(x))

        # функция для получение матрицы близости векторов
        def get_sim_matrix(df, thrash = 0.909):
            array_vectors = df['vectors'].values
            array_vectors_lengths = df['vec_length']
            
            cos_sim = (np.stack(array_vectors,axis = 0) @ np.stack(array_vectors,axis = 1)) / (np.outer(array_vectors_lengths, array_vectors_lengths))
            for i in range(cos_sim.shape[0]):
                cos_sim[i][i] = 0
                
            best_cos_sims = np.where(cos_sim > thrash, cos_sim,int(0))
            
            return best_cos_sims
        
        # матрица близости
        self.similarity_matrix = get_sim_matrix(self.df)
        n = self.similarity_matrix.shape[0]

        indices_to_keep = list(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if self.similarity_matrix[i, j] > 0.9:
                    if j in indices_to_keep:
                        self.df = self.df.drop(
                            index=self.indexes[j],
                            axis=0
                        )

        return self.df