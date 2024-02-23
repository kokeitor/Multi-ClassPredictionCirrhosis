### Cross validation strategy implementation
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import cross_validate,KFold,StratifiedKFold
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from typing import List,Dict,Optional
from sklearn.pipeline import Pipeline

def cv_function( 
                    X_train :  pd.DataFrame ,
                    y_train :  pd.Series , 
                    pipelines : List[Pipeline] = [], 
                    n_splits : int = 10,
                    metrics : List[str] = [],
                    cv_strategies: List[str] = [] 
                    
                ) -> Dict[str,pd.DataFrame]:
    """
    Funcion para comparar diferentes cv stratategies aplicando diferentes metricas para un mismo pipeline

    Parametros
    ----------
        - pipelines : List[Pipeline] | Lista de objetos Pipeline a evaluar
        - metrics : List[str] | Nombres de las "function scoring" de la biblioteca sklearn 
                              | Posibles valores : ["neg_mean_absolute_error","neg_root_mean_squared_error","r2", ...]
        - cv_strategies : List[str] | Nombres de las estrategias de validacion cruzada
                                    | Posibles valores : ["k-folds","Stratified K-folds","TimeSeriesSplit",...]
        - n_splits : int = 10 | numero de splits para estrategia de validacion 
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
            
            for pipe_indx , pipe in enumerate(pipelines):
                
                # Dataframe columns names
                train_column_names = [f"Val {score}" for score in metrics]
                val_column_names = [f"Train {score}" for score in metrics]
                test_scores_column_names = [f"Test {score}" for score in metrics]
                
                # Initializing score Dataframe
                test_scores = pd.DataFrame(
                                        index =  [str(pipe.__class__).split('.')[-1][0:len(str(pipe.__class__).split('.')[-1])-2]] + train_column_names + val_column_names + test_scores_column_names,
                                        columns = [cv_s for cv_s in cv_strategies],
                                        )

                for _ , cv_name in enumerate(cv_strategies):
                    
                    cv = cv_strategies_dict.get(cv_name)
                    
                    if cv != None:
                        cv_dict = cross_validate(   
                                                    pipe, 
                                                    X_train, 
                                                    y_train, 
                                                    return_estimator = True,
                                                    return_train_score = True,
                                                    scoring = metrics, 
                                                    cv = cv,
                                                    error_score = 'raise',
                                                    n_jobs = -1
                                                )
                        

                        for score in metrics:
                            
                            # Filling dataframe
                            test_scores.loc[f"Val {score}" ,cv_name] = np.mean(cv_dict[f"test_{score}"])
                            test_scores.loc[f"Train {score}" ,cv_name] = np.mean(cv_dict[f"train_{score}"])
                            
                    else: 
                        print(f"Error : {cv_name} strategy is not defined")
                        
                        
                # Introduccion de df informativo dentro de dict
                dict_scores[f"Pipeline : {pipe_indx}"] = test_scores

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