from base_am.metodo import MetodoAprendizadoDeMaquina
import pandas as pd
from .preprocessamento_atributos_competicao import gerar_atributos_letra_musica
from base_am.resultado import Resultado
from typing import Union, List
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
class ClasseNumerica:
    
    def __init__(self):
        self.dic_int_to_nom_classe = {}
        self.dic_nom_classe_to_int = {}

    def class_to_number(self,y:List[str]) -> List[int]:
        arr_int_y = []

        #mapeia cada classe para um número
        for rotulo_classe in y:
            #cria um número para esse rotulo de classe, caso não exista ainda
            if rotulo_classe not in self.dic_nom_classe_to_int:
                int_new_val_classe = len(self.dic_nom_classe_to_int.keys())
                self.dic_nom_classe_to_int[rotulo_classe] = int_new_val_classe
                self.dic_int_to_nom_classe[int_new_val_classe] = rotulo_classe

            #adiciona esse item
            arr_int_y.append(self.dic_nom_classe_to_int[rotulo_classe])

        return arr_int_y
    
class MetodoCompeticaoHierarquico(MetodoAprendizadoDeMaquina):
    #você pode mudar a assinatura desta classe (por exemplo, usar dois metodos e o resultado da predição
    # seria a combinação desses dois)
    def __init__(self,ml_method:Union[ClassifierMixin,RegressorMixin], col_classe_prim_nivel="",max_df=0.9):
        #caso fosse vários métodos, não há problema algum passar um array de todos os métodos como parametro ;)
        self.ml_method = ml_method
        self.col_classe_prim_nivel = col_classe_prim_nivel

        self.obj_class_prim_nivel = ClasseNumerica()
        self.obj_class_final = ClasseNumerica()
        self.obj_class_seg_nivel = {}
        self.result_prim_nivel = None
        self.max_df = max_df
        
    def filtrar_por_agrupamento_prim_nivel(self, agrupamento, arr_predict_grupo):
        arr_pos = []
        for pos,value_genre in enumerate(arr_predict_grupo):
            if self.obj_class_prim_nivel.dic_int_to_nom_classe[value_genre] == agrupamento:
                arr_pos.append(pos)
        return arr_pos

    def obtem_y(self, df_treino:pd.DataFrame, df_data_to_predict:pd.DataFrame, col_classe:str,prim_nivel:bool, agrupamento:str=""):
        
        col_y_atual = self.col_classe_prim_nivel if prim_nivel else  col_classe
        y_to_predict = None
        
        #converte no treino e teste
        if prim_nivel:
            y_treino = self.obj_class_prim_nivel.class_to_number(df_treino[col_y_atual])
            #y_to_predict pod não existir (no dataset de teste fornecido pelo professor, por ex)
            if col_classe in df_data_to_predict.columns:
                y_to_predict = self.obj_class_prim_nivel.class_to_number(df_data_to_predict[col_y_atual])
        else:
            y_treino = self.obj_class_seg_nivel[agrupamento].class_to_number(df_treino[col_y_atual])
            if col_classe in df_data_to_predict.columns:
                y_to_predict = self.obj_class_seg_nivel[agrupamento].class_to_number(df_data_to_predict[col_y_atual])

        return y_treino,y_to_predict

    def obtem_x(self, df_treino:pd.DataFrame, df_data_to_predict:pd.DataFrame, col_classe:str):
        x_treino = df_treino.drop(["id", col_classe, self.col_classe_prim_nivel] , axis = 1)
        x_to_predict = df_data_to_predict

        if col_classe in df_data_to_predict.columns:
            x_to_predict = df_data_to_predict.drop(["id", col_classe, self.col_classe_prim_nivel], axis = 1)
        return x_treino, x_to_predict



    def eval(self, df_treino:pd.DataFrame, df_data_to_predict:pd.DataFrame, col_classe:str, seed:int=1):
        self.obj_class_final.class_to_number(df_treino[col_classe])
        # print(f"Mapeamento geral: {self.obj_class_final.dic_int_to_nom_classe}")
        df_treino_bow = []
        df_to_predict_bow = []
        #################### Primeiro Nivel #################################                                                    
        #separação da classe 
        x_treino, x_to_predict = self.obtem_x(df_treino, df_data_to_predict, col_classe)
        y_treino, y_to_predict = self.obtem_y(df_treino, df_data_to_predict, col_classe, True)
        # print(f"x_treino: {y_treino} y_to_predict:{y_to_predict}")
        
        #geração dos atributos
        x_treino_bow, x_to_predict_bow = gerar_atributos_letra_musica(x_treino, x_to_predict, self.max_df)
        
        #geração do modelo e predicçaõ do primeiro nivel
        self.ml_method.fit(x_treino_bow, y_treino)
        arr_predict_prim_nivel = self.ml_method.predict(x_to_predict_bow)
        
        self.result_prim_nivel = Resultado(y_to_predict, arr_predict_prim_nivel)
        
        # print(f"Predict Primeiro nivel: {arr_predict_prim_nivel}")
        ################### Segundo nivel  ##########################
        arr_predict_final = [None for i in range(len(arr_predict_prim_nivel))]
        #no dataset fonecido pelo professor, a col_classe não existe
        y_to_predict_final = None
        if col_classe in df_data_to_predict.columns:
            y_to_predict_final = self.obj_class_final.class_to_number(df_data_to_predict[col_classe])

        #para cada classe do treino
        for agrupamento in df_treino[self.col_classe_prim_nivel].unique(): 
            # print(f"Agrupamento: {agrupamento}")

            #usa o segundo nivel apenas nos agrupamentos que efetivamente possuem mais de uma classe no segundo nivel
            df_treino_grupo = df_treino[df_treino[self.col_classe_prim_nivel]==agrupamento]
            arr_pos_predict = self.filtrar_por_agrupamento_prim_nivel(agrupamento, arr_predict_prim_nivel)

            if len(arr_pos_predict)==0:
                continue
                
            #col_classe => col_classe do seg nivel
            if len(df_treino_grupo[col_classe].unique()) > 1:
                df_data_to_predict_grupo = df_data_to_predict.iloc[arr_pos_predict]
                
                self.obj_class_seg_nivel[agrupamento] = ClasseNumerica()
                x_treino_grupo, x_to_predict_grupo = self.obtem_x(df_treino_grupo, df_data_to_predict_grupo, col_classe)
                y_treino_grupo, y_to_predict_grupo = self.obtem_y(df_treino_grupo, df_data_to_predict_grupo, col_classe, False, agrupamento)


                x_treino_grupo_bow, x_to_predict_grupo_bow = gerar_atributos_letra_musica(x_treino_grupo, x_to_predict_grupo, self.max_df)


                # print(self.obj_class_seg_nivel[agrupamento].dic_int_to_nom_classe)
                # print(f"y treino: {y_treino_grupo} to predict: {y_to_predict_grupo}") 
                self.ml_method.fit(x_treino_grupo_bow, y_treino_grupo)
                arr_predict_grupo = self.ml_method.predict(x_to_predict_grupo_bow)
                # print(f"Predições: {arr_predict_grupo}")                                            
                for pos_grupo,val_predict_grupo in enumerate(arr_predict_grupo):
                    pos_original = arr_pos_predict[pos_grupo]
                    # print(f"Posicao {pos_grupo} correspond a posicao {pos_original}")
                    final_classe_nome = self.obj_class_seg_nivel[agrupamento].dic_int_to_nom_classe[val_predict_grupo]
                    
                    arr_predict_final[pos_original] = self.obj_class_final.dic_nom_classe_to_int[final_classe_nome]
            
            else:
                for pos,pos_original in enumerate(arr_pos_predict):
                    # print(pos_original)
                    val_predict = arr_predict_prim_nivel[pos_original]
                    final_classe_nome = self.obj_class_prim_nivel.dic_int_to_nom_classe[val_predict]
                    arr_predict_final[pos_original] = self.obj_class_final.dic_nom_classe_to_int[final_classe_nome]
        # print(y_to_predict, arr_predict_final)
        # print("jasdiajsaiajdisajidjsa")

        # print(arr_predict_final)
        return Resultado(y_to_predict_final, arr_predict_final)
