import pandas as pd
from base_am.preprocessamento_atributos import BagOfWords, BagOfItems
from sklearn.feature_extraction.text import TfidfVectorizer

def gerar_atributos_letra_musica(df_treino:pd.DataFrame, df_data_to_predict: pd.DataFrame, max_df:float) -> pd.DataFrame:
    bow_amostra = BagOfWordsLyrics(max_df)
    df_bow_treino = bow_amostra.cria_bow(df_treino,"lyrics")
    df_bow_data_to_predict = bow_amostra.aplica_bow(df_data_to_predict,"lyrics")

    return df_bow_treino,df_bow_data_to_predict

stop_list = {"i","he","she","it","a","the","almost","do","does"}
class BagOfWordsLyrics(BagOfWords):
    def __init__(self, max_df:float):
        #O TfidfVectorizer que é resposavel por gerar a representação BOW
        #você pode mudar a parametrização do mesmo (inclusive, na fase de avaliação)
        #norm: normalização para que todos os valores fiquem entre 0 e 1
        #max_df: remove palavras que ocorrem em mais que 90% dos documentos
        #stop_words: lista das stopwords a serem removidas
        self.vectorizer = TfidfVectorizer(norm="l2",max_df=max_df, stop_words=stop_list)


