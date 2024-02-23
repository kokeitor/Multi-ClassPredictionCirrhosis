### Cross validation strategy implementation
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import cross_validate,KFold,StratifiedKFold
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from typing import List,Dict
from sklearn.pipeline import Pipeline

def cv_function( 
                    X :  pd.DataFrame ,
                    y :  pd.Series , 
                    pipelines : List[Pipeline] = [], 
                    n_splits : int = 10,
                    metrics : List[str] = [],
                    cv_strategies: List[str] = [] 
                    
                ) -> Dict[pd.DataFrame]:
    """
    Funcion para comparar diferentes cv stratategies aplicando diferentes metricas para un mismo pipeline

    Parametros
    ----------
        - pipelines : List[Pipeline] | Lista de objetos Pipeline a evaluar
        - metrics : List[str] | Nombres de las "function scoring" de la biblioteca sklearn 
                              | Posibles valores : ["neg_mean_absolute_error","neg_root_mean_squared_error","r2", ...]
        - cv_strategies : List[str] | Nombres de las estrategias de validacion cruzada
                                    | Posibles valores : ["k-folds","Stratified K-folds","TimeSeriesSplit",...]
        - x : pd.DataFrame | X set
        - y : pd.DataFrame | y set 

    Retorna
    -------
        - dict_scores : dict | Diccionario con los scores de validacion

    """

    # Estrategias de validacion cruzada
    kf = KFold(n_splits = n_splits)
    skf = StratifiedKFold(n_splits = n_splits)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_strategies_dict = {
                            "k-folds" : kf,
                            "Stratified K-folds" : skf,
                            "TimeSeriesSplit" : tscv
                        }

    if cv_strategies != [] and metrics != []:
        
        if pipelines != []:
            dict_scores =  {}
            
            for _ , pipe in enumerate(pipelines):

                # Initializing info dataframe
                test_scores = pd.DataFrame(
                                        index = [cv_s for cv_s in cv_strategies],
                                        columns = [f"test {score} mean" for score in metrics] ,
                                        )

                for _ , cv_name in enumerate(cv_strategies):
                    
                    cv = cv_strategies_dict.get(cv_name)
                    
                    if cv != None:
                        cv_dict = cross_validate(   
                                                    pipe, 
                                                    X, 
                                                    y, 
                                                    return_estimator = True,
                                                    scoring = metrics, 
                                                    cv = cv,
                                                    error_score = 'raise',
                                                    n_jobs = -1
                                                )

                        for score in metrics:
                            
                            # Filling dataframe
                            test_scores.loc[cv_name , f"validation {score} mean" ] = np.mean(cv_dict[f"test_{score}"])
                    else: 
                        print(f"Error : {cv_name} strategy is not defined")
                        
                        
                # Introduccion de df informativo dentro de dict
                dict_scores[f"Pipeline for {cv_dict["estimator"]}"] = test_scores

            return dict_scores
        else:
            print("Error : No Pipelines provided")
    else:
        print("Error : Cross-validation strategies and Score metrics are not defined")



"""
# Function call
regression_metrics = ["neg_mean_absolute_error","neg_root_mean_squared_error","r2"]
dict_scoring = cv_function( pipeline  = pipeline_1, x = X_train , y  = y_train , metrics  = regression_metrics)
# cv function results
for pipe_index,pipe_scores in dict_scoring.values():
  print(f"Para el pipeline numero {pipe_index+1}: ")
  print(pipe_scores.head())
"""