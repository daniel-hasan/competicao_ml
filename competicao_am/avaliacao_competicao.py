from base_am.avaliacao import OtimizacaoObjetivo
from base_am.metodo import MetodoAprendizadoDeMaquina
from base_am.resultado import Fold, Resultado
from competicao_am.metodo_competicao import MetodoCompeticaoHierarquico
import optuna
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


class OtimizacaoObjetivoSVMCompeticao(OtimizacaoObjetivo):
    def __init__(self, fold:Fold, num_arvores_max:int=5):
        super().__init__(fold)
        self.num_arvores_max = num_arvores_max

    def obtem_metodo(self,trial: optuna.Trial)->MetodoAprendizadoDeMaquina:
        #Um custo adequado para custo pode variar muito, por ex, para uma tarefa 
        #o valor de custo pode ser 10, para outra, 32000. 
        #Assim, normalmente, para conseguir valores mais distintos,
        #usamos c=2^exp_cost
        exp_cost = trial.suggest_uniform('exp_cost', 0, 3) 

        scikit_method = LinearSVC(C=2**exp_cost, random_state=2, class_weight='balanced')

        return MetodoCompeticaoHierarquico(scikit_method, "grouped_genre")

    def resultado_metrica_otimizacao(self,resultado: Resultado) -> float:
        return resultado.macro_f1
