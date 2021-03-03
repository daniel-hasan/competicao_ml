from competicao_am.metodo_competicao import MetodoCompeticaoHierarquico
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np

def gerar_saida_teste( df_data_to_predict, col_classe, num_grupo):
    """
    Assim como os demais códigos da pasta "competicao_am", esta função 
    só poderá ser modificada na fase de geração da solução. 
    """
    np.random.seed(2)
    scikit_method = LinearSVC(C=7, random_state=2)
    ml_method = MetodoCompeticaoHierarquico(scikit_method,"grouped_genre")
    
    #o treino será sempre o dataset completo - sem nenhum dado a mais e sem nenhum preprocessamento
    #a função gerar_saida_teste e as que você irá invocar que deve encarregar de fazer o preprocessamento
    df_treino = pd.read_csv("datasets/lyrics_amostra.csv")


    #gera as representações e seu resultado
    result = ml_method.eval(df_treino, df_data_to_predict, col_classe)


    #grava o resultado obtido
    with open(f"predict_grupo_{num_grupo}.txt","w") as file_predict:
        for predict in result.predict_y:
            file_predict.write(ml_method.obj_class_final.dic_int_to_nom_classe[predict]+"\n")
