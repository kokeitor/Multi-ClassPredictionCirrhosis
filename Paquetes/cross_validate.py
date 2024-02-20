### Cross validation strategy implementation
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import cross_validate,KFold,StratifiedKFold
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score



def cv_function( pipelines : list = [], x :  pd.DataFrame = None , y :  pd.DataFrame = None , metrics : list[str] = []) -> dict:
    """
    Funcion para comparar diferentes cv stratategies aplicando diferentes metricas para un mismo pipeline

    Parametros
    ----------
    - pipelines: (list)
    - metrics : (list[str]) nombres de las "function scoring" de sklearn
    - x
    - y

    Retorna
    -------
    - dict_scores (dict)

    """

    # Estrategias de validacion cruzada
    n_splits = 5
    kf = KFold(n_splits = n_splits)
    skf = StratifiedKFold(n_splits = n_splits)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_strategies = [
                ("k-folds",kf),
                ("TimeSeriesSplit",tscv)
                ]
    if pipelines != []:
        dict_scores =  {}
        for pipe_index , pipe in enumerate(pipelines):

            # Initializing info dataframe
            test_scores = pd.DataFrame(
                                    index = [cv_s[0] for cv_s in cv_strategies],
                                    columns = [f"test {score} mean" for score in metrics] ,
                                    )

            for s_index , s in enumerate(cv_strategies):

                cv_str_name  = s[0]
                cv_str_object = s[1]

                cv_dict = cross_validate(pipe, x, y, scoring = metrics, cv = cv_str_object ,error_score='raise')

                for score in metrics:
                    # Filling dataframe
                    test_scores.loc[cv_str_name , f"test {score} mean" ] = np.mean(cv_dict[f"test_{score}"])
                
                # Introduccion de df informativo dentro de dict
                dict_scores[f"Pipeline {pipe_index+1}"] = test_scores

        return dict_scores



"""
# Function call
regression_metrics = ["neg_mean_absolute_error","neg_root_mean_squared_error","r2"]
dict_scoring = cv_function( pipeline  = pipeline_1, x = X_train , y  = y_train , metrics  = regression_metrics)
# cv function results
for pipe_index,pipe_scores in dict_scoring.values():
  print(f"Para el pipeline numero {pipe_index+1}: ")
  print(pipe_scores.head())
"""