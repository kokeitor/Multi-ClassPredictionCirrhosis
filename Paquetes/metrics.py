import pandas as pd
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from typing import  Callable, Optional, List, Dict, Tuple
from Paquetes.optimization import execution_time

# Metrics for clasification problems 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve,confusion_matrix,jaccard_score
from sklearn.metrics import confusion_matrix

@execution_time
def clasification_metrics(
                            y_train : np.ndarray,
                            x_train : np.ndarray ,
                            clasifier : List[object],
                            average : Optional[str] = None,
                            y_test : np.ndarray = np.zeros(shape = 0),
                            x_test : np.ndarray = np.zeros(shape = 0),
                            metrics : List[str] = ["Accuracy"],
                            plot_roc_curve  : bool = False,
                            plot_confusion_matrix  : bool = False,
                            class_names : List[str] = []
                        ) -> List[pd.DataFrame]:
    """
    Devuelve dataframes para cada clasifier con las metricas especificadas y aplicadas a las prediciones (por defecto, si no varia el valor del aprametro "average" la funcion
    toma el clasificador como binario). Capaz de aplicar la predicion (y logiacamente las métricas) al conjunto de test y train.
    Por defecto a no ser que el parametro 'average' se inicialize o se establezca como in argument debe pasarse un clasificador entrenado y binario 
    es decir, y_pred debe contener 0 y 1. Si 'average' se define, se pueden usar varios criterios para el caluclo de metricas para problema multiclase (micro,macro,ponderado,etc).
    En este caso el claifioer tambien debe ser entrenado y debe ser un clasifier multiclase.
    Si 'average' se define no estan disponibles todas las metricas del caso binario, por ejemplo: no se plotea la ROC curve, dado que habria una ROC curve por target label.
    
    Paramaters
    ----------
        - y_train : np.ndarray 
                    Real target train labels
        - x_train : np.ndarray 
        - clasifier : List[object]
        - average : Optional[str]
                    ['micro','macro','samples', 'weighted','binary'== ""]
        - y_test : np.ndarray
                    Real target test labels
        - x_test : np.ndarray 
        - metrics : List[str]
                    Default "Accuracy"
        - plot_roc_curve: bool
                        plot de roc curve (and area under the curve). Note: only apply to binary clasification and test set
        - plot_confusion_matrix  : bool 
                                    Note: only apply to binary clasification and test y train sets
        - class_names : List[str] 
    Return
    ------
        - List[pd.DataFrame] : 
        Lista de df con las metricas en columnas y los dataset donde las aplica en filas (train y test)

    """
    # Multilabel clasification metrics 8 average != None o binary)
    if average != None:
        # Loop for all the passed predictors
        clasifiers_metrics = []
        for clsf in clasifier:

            # Dict to map the input str arguments to the score/metric objects
            scores_mapping = {
                                    "Recall" : recall_score(y_true = y_train, y_pred = clsf.predict(x_train),average= average ),
                                    "Precision": precision_score(y_true = y_train, y_pred = clsf.predict(x_train), average= average),
                                    "F1Score": f1_score(y_true = y_train, y_pred = clsf.predict(x_train), average= average),
                                    "JaccardIndex" :  jaccard_score(y_true = y_train, y_pred = clsf.predict(x_train),average= average)

                            }
            
            # Datafrane creation 
            df_results = pd.DataFrame( 
                                        data = [],
                                        columns = [k for k in scores_mapping.keys()],
                                        index = ["Train", "Test"]
                                )

            # For test set if it is defined
            metrics_test_aux = []
            if y_test.shape[0] != 0 and x_test.shape[0] != 0:

                # Dict to map the input str arguments to the score/metric objects
                scores_mapping_test = {
                                    "Recall" : recall_score(y_true = y_test, y_pred = clsf.predict(x_test),average= average ),
                                    "Precision": precision_score(y_true = y_test, y_pred = clsf.predict(x_test), average= average),
                                    "F1Score": f1_score(y_true = y_test, y_pred = clsf.predict(x_test), average= average),
                                    "JaccardIndex" :  jaccard_score(y_true = y_test, y_pred = clsf.predict(x_test),average= average)
                                        }
            
                # Mapping the str metrics with the values (obj metrics) in the dictionary using: List Comprehension
                if metrics != None:
                    metrics_test_aux = [[(k,v) for k ,v in scores_mapping_test.items() if k == m] for m in metrics if m in scores_mapping_test]
                    metrics_test_aux = ([v[0] for _ ,v in enumerate(metrics_test_aux)]) # transformar doble lista de tuplas en lista unica de tuplas
                

            # Mapping the str metrics with the values (obj metrics) in the dictionary using: List Comprehension
            metrics_aux = []
            if metrics != None:
                metrics_aux = [[(k,v) for k ,v in scores_mapping.items() if k == m] for m in metrics if m in scores_mapping]
                metrics_aux = ([v[0] for _ ,v in enumerate(metrics_aux)])# transformar doble lista de tuplas en lista unica de tuplas

            # Filling the df
            for key,value in metrics_aux:
                df_results.loc["Train",f"{key}"] = value
            if metrics_test_aux != []:
                for key,value in metrics_test_aux:
                    df_results.loc["Test",f"{key}"] = value

            clasifiers_metrics.append(df_results)
            
    # Binary clasification metrics (average no se ha definido y por defecto es == 'binary' para las funciones que calculan las metricas)
    else:
        # Loop for all the passed predictors
        clasifiers_metrics = []
        for clsf in clasifier:
            # Calculate confusion matrix
            tn,fp,fn,tp = confusion_matrix(y_true = y_train,  y_pred = clsf.predict(x_train)).ravel()
            
            # Dict to map the input str arguments to the score/metric objects
            scores_mapping = {
                                "Accuracy": accuracy_score(y_true = y_train, y_pred = clsf.predict(x_train)),
                                "Recall" : recall_score(y_true = y_train, y_pred = clsf.predict(x_train)),
                                "Precision": precision_score(y_true = y_train, y_pred = clsf.predict(x_train)),
                                "F1Score": f1_score(y_true = y_train, y_pred = clsf.predict(x_train)),
                                "RocCurveArea": roc_auc_score(y_true = y_train, y_score = clsf.predict(x_train)),
                                "TN": tn, 
                                "FP":fp , 
                                "FN":fn, 
                                "TP" :tp ,
                                "Especificidad" :  tn / tn + tp,
                                "JaccardIndex" :  jaccard_score(y_true = y_train, y_pred = clsf.predict(x_train))
                            }
            
            # Datafrane creation 
            df_results = pd.DataFrame( 
                                        data = [],
                                        columns = [k for k in scores_mapping.keys()],
                                        index = ["Train", "Test"]
                                )
            # For test set if it is defined
            metrics_test_aux = []
            if y_test.shape[0] != 0 and x_test.shape[0] != 0:
                # Calculate confusion matrix
                tn_test,fp_test,fn_test,tp_test = confusion_matrix(y_true = y_test,  y_pred = clsf.predict(x_test)).ravel()
                
                # Dict to map the input str arguments to the score/metric objects
                scores_mapping_test = {
                                        "Accuracy": accuracy_score(y_true = y_test, y_pred = clsf.predict(x_test)),
                                        "Recall" : recall_score(y_true = y_test, y_pred = clsf.predict(x_test)),
                                        "Precision": precision_score(y_true = y_test, y_pred = clsf.predict(x_test)),
                                        "F1Score": f1_score(y_true = y_test, y_pred = clsf.predict(x_test)),
                                        "RocCurveArea": roc_auc_score(y_true = y_test, y_score = clsf.predict(x_test)),
                                        "TN": tn_test, 
                                        "FP":fp_test, 
                                        "FN":fn_test, 
                                        "TP" :tp_test ,
                                        "Especificidad" :  tn_test / tn_test + tp_test,
                                        "JaccardIndex" :  jaccard_score(y_true = y_test, y_pred = clsf.predict(x_test))
                                        }
            
                # Mapping the str metrics with the values (obj metrics) in the dictionary using: List Comprehension
                if metrics != None:
                    metrics_test_aux = [[(k,v) for k ,v in scores_mapping_test.items() if k == m] for m in metrics if m in scores_mapping_test]
                    metrics_test_aux = ([v[0] for _ ,v in enumerate(metrics_test_aux)]) # transformar doble lista de tuplas en lista unica de tuplas
                

            # Mapping the str metrics with the values (obj metrics) in the dictionary using: List Comprehension
            metrics_aux = []
            if metrics != None:
                metrics_aux = [[(k,v) for k ,v in scores_mapping.items() if k == m] for m in metrics if m in scores_mapping]
                metrics_aux = ([v[0] for _ ,v in enumerate(metrics_aux)])# transformar doble lista de tuplas en lista unica de tuplas

            # Filling the df
            for key,value in metrics_aux:
                df_results.loc["Train",f"{key}"] = value
            if metrics_test_aux != []:
                for key,value in metrics_test_aux:
                    df_results.loc["Test",f"{key}"] = value

            clasifiers_metrics.append(df_results)

            # plotting roc curve (only x train set)
            if plot_roc_curve:

                fpr, tpr, _ = roc_curve(y_train, clsf.predict(x_train))

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                                        x=fpr, 
                                        y=tpr,
                                        mode="lines+markers",
                                        line=dict(color="blue"),
                                        marker=dict(size=8),
                                        name= f"ROC curve"
                                        ))

                # Y, si quisiéramos añadir el área bajo la curva:
                auc = roc_auc_score(y_test, clsf.predict(x_test))
                fig.add_trace(go.Scatter(
                                        x=fpr, 
                                        y=tpr,
                                        mode="lines+markers",
                                        line=dict(color="blue"),
                                        marker=dict(size=8),
                                        fill="tozeroy",# Para rellenar el gráfico
                                        fillcolor="rgba(100, 0, 255, 0.3)" ,
                                        name = f"[AUC-ROC = {auc:,.2f}]"
                                        ))
                
                fig.update_layout(
                                title=f"ROC curve - {clsf}",
                                xaxis = dict(
                                                title = "Tasa de Falsos Positivos (FPR)",
                                                autorange = True,
                                                showline = None,
                                                showgrid = None,
                                                gridcolor = None,
                                                showticklabels = True,
                                                zeroline = False,
                                                linecolor = None,
                                                linewidth = 0.5,
                                                ticks = 'outside',
                                                tickfont = dict(
                                                                family = 'Arial',
                                                                color = 'rgb(82,82,82)',
                                                                size = 12
                                                                )
                                            ),
                                yaxis = dict(
                                title = "Tasa de Verdaderos Positivos (TPR)",
                                autorange = True,
                                showline = None,
                                showgrid = None,
                                gridcolor = None,
                                showticklabels = True,
                                zeroline = False,
                                linecolor = None,
                                linewidth = 0.5,
                                ticks = 'outside',
                                tickfont = dict(
                                                family = 'Arial',
                                                color = 'rgb(82,82,82)',
                                                size = 12
                                                )),
                                autosize = True,
                                width=900,
                                height = 700,
                                margin = dict(
                                                autoexpand = True,
                                                l= 70,
                                                r = 120,
                                                t = 100,
                                                b = 60
                                                ),
                                showlegend = True,
                                plot_bgcolor = None,
                                legend = dict(
                                                bgcolor = 'white',
                                                bordercolor = 'black',
                                                borderwidth = 0.5,
                                                title = dict(
                                                            font = dict(
                                                                        family = 'Arial',
                                                                        color = 'black',
                                                                        size = 16
                                                                        ),
                                                            side = 'top'
                                                            ),
                                                font = dict(
                                                            family = 'Arial',
                                                            color = 'rgb(82,82,82)',
                                                            size = 12
                                                            )

                                                ))
                fig.show()

            # Plotting confusion matrix (only train set)
            if plot_confusion_matrix:

                # Calculate confusion matrix for x train
                c_matrix = confusion_matrix(y_true = y_train,  y_pred = clsf.predict(x_train))
                class_names = class_names if class_names != [] else ["0","1"]

                # Crear la matriz de confusión usando Plotly
                fig = ff.create_annotated_heatmap(
                                                z = c_matrix, 
                                                colorscale = 'blues',
                                                x = class_names, 
                                                y = class_names,
                                                annotation_text = c_matrix.astype(str))
                # Añadir títulos y etiquetas
                fig.update_layout(
                                title=f'Train Confusion matrix - {clsf}',
                                xaxis = dict(
                                                title='Predicted label',
                                                tickmode='array',
                                                tickvals=[0, 1],
                                                autorange = True,
                                                showline = None,
                                                showgrid = True,
                                                gridcolor = None,
                                                showticklabels = True,
                                                zeroline = False,
                                                linecolor = None,
                                                linewidth = 0.5,
                                                ticks = 'outside',
                                                tickfont = dict(
                                                                family = 'Arial',
                                                                color = 'rgb(82,82,82)',
                                                                size = 12
                                                            )
                                                        ),
                                yaxis = dict(
                                title='True label',
                                tickmode='array',
                                tickvals=[0, 1],
                                autorange='reversed',
                                showline = None,
                                showgrid = None,
                                gridcolor = None,
                                showticklabels = True,
                                zeroline = False,
                                linecolor = None,
                                linewidth = 0.5,
                                ticks = 'outside',
                                tickfont = dict(
                                                family = 'Arial',
                                                color = 'rgb(82,82,82)',
                                                size = 12
                                                )),
                                autosize = False,
                                width=900,
                                height = 700,
                                margin = dict(
                                                autoexpand = True,
                                                l= 70,
                                                r = 120,
                                                t = 120,
                                                b = 60
                                                ),
                                showlegend = False,
                                plot_bgcolor = None,
                                legend = dict(
                                                bgcolor = 'white',
                                                bordercolor = 'black',
                                                borderwidth = 0.5,
                                                title = dict(
                                                            font = dict(
                                                                        family = 'Arial',
                                                                        color = 'black',
                                                                        size = 16
                                                                        ),
                                                            side = 'top'
                                                            ),
                                                font = dict(
                                                            family = 'Arial',
                                                            color = 'rgb(82,82,82)',
                                                            size = 12
                                                            )

                                                ))
                                

                                
                # Mostrar el gráfico
                fig.show()
                
            # Plotting confusion matrix (only test set)
            if plot_confusion_matrix:

                # Calculate confusion matrix for x train
                c_matrix = confusion_matrix(y_true = y_test,  y_pred = clsf.predict(x_test))
                class_names = class_names if class_names != [] else ["0","1"]

                # Crear la matriz de confusión usando Plotly
                fig = ff.create_annotated_heatmap(
                                                z = c_matrix, 
                                                colorscale = 'reds',
                                                x = class_names, 
                                                y = class_names,
                                                annotation_text = c_matrix.astype(str))
                # Añadir títulos y etiquetas
                fig.update_layout(
                                title=f'Test Confusion matrix - {clsf}',
                                xaxis = dict(
                                                    title='Predicted label',
                                                    tickmode='array',
                                                    tickvals=[0, 1],
                                                    autorange = True,
                                                    showline = None,
                                                    showgrid = True,
                                                    gridcolor = None,
                                                    showticklabels = True,
                                                    zeroline = False,
                                                    linecolor = None,
                                                    linewidth = 0.5,
                                                    ticks = 'outside',
                                                    tickfont = dict(
                                                                    family = 'Arial',
                                                                    color = 'rgb(82,82,82)',
                                                                    size = 12
                                                                    )
                                            ),
                                yaxis = dict(
                                title='True label',
                                tickmode='array',
                                tickvals=[0, 1],
                                autorange='reversed',
                                showline = None,
                                showgrid = None,
                                gridcolor = None,
                                showticklabels = True,
                                zeroline = False,
                                linecolor = None,
                                linewidth = 0.5,
                                ticks = 'outside',
                                tickfont = dict(
                                                family = 'Arial',
                                                color = 'rgb(82,82,82)',
                                                size = 12
                                                )),
                                autosize = False,
                                width=900,
                                height = 700,
                                margin = dict(
                                                autoexpand = False,
                                                l= 70,
                                                r = 120,
                                                t = 120,
                                                b = 60
                                                ),
                                showlegend = False,
                                plot_bgcolor = None,
                                legend = dict(
                                                bgcolor = 'white',
                                                bordercolor = 'black',
                                                borderwidth = 0.5,
                                                title = dict(
                                                            font = dict(
                                                                        family = 'Arial',
                                                                        color = 'black',
                                                                        size = 16
                                                                        ),
                                                            side = 'top'
                                                            ),
                                                font = dict(
                                                            family = 'Arial',
                                                            color = 'rgb(82,82,82)',
                                                            size = 12
                                                            )

                                                ))

                                
                # Mostrar el gráfico
                fig.show()

    return clasifiers_metrics

"""
Ejemplo de uso multiclase: (avergae !=None)
<lista de df con resultados > = (clasification_metrics(
                                                            y_train = y_train,
                                                            x_train= X_train_std,
                                                            y_test  = np.zeros(0), # ó y_test
                                                            x_test = np.zeros(0), # ó x_test
                                                            clasifier = [tree,svm,knn],
                                                            average = 'micro',
                                                            metrics = [
                                                                            "Recall" ,
                                                                            "Precision",
                                                                            "F1Score",
                                                                            "Especificidad" ,
                                                                            "JaccardIndex" 
                                                                        ],
                                                            plot_roc_curve = True,
                                                            plot_confusion_matrix = True,
                                                            class_names = ["0","1"]

        ))
Ejemplo de uso binario: (avergae =None o no se define como parametro)
<lista de df con resultados > = (clasification_metrics(
                                                            y_train = y_train,
                                                            x_train= X_train_std,
                                                            y_test  = np.zeros(0), # ó y_test
                                                            x_test = np.zeros(0), # ó x_test
                                                            clasifier = [tree,svm,knn],
                                                            metrics = [
                                                                        "Accuracy",
                                                                        "Recall" ,
                                                                        "Precision",
                                                                        "F1Score",
                                                                        "RocCurveArea",
                                                                        "TN",
                                                                        "FP",
                                                                        "FN",
                                                                        "TP" ,
                                                                        "Especificidad" ,
                                                                        "JaccardIndex" 
                                                                        ],
                                                            plot_roc_curve = True,
                                                            plot_confusion_matrix = True,
                                                            class_names = ["0","1"]

        ))
"""