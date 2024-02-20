import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import  Callable
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from itertools import product
from typing import  Callable, Optional, List, Dict, Tuple


def pca(data: pd.DataFrame = None, encodings: list[Callable] = [], num_transformers : list[Callable] = [], pca_components: list[int] = [1]) -> pd.DataFrame:

  """
  Funcion que aplica PCA tras codificar las columnas categoricas y aplicar una transformacion (o no) sobre todas las columnas numericas del df (tras la codificacion todas seran numericas).
  Despues retorna un dataframe con informacion del PCA realizado y crea graficas que tambien lo describen

  Parametros
  ----------
    key word arguments:
      - data : (pd.DataFrame) Dataframe sobre el que se quiere llevar acabo el analisis de dimensionalidad
      - encodings : (list[Callable]) Lista con los codificadores de las columnas categoricas que se quieren aplicar
      - num_transformers : (list[Callable])  Lista con los numerical transformers que se quieren aplicar sobre todo el df tras la codificacion de las columnas categoricas
      - pca_components: (list[int]) lista con el numero de pca components que se quieren aplicar en el estudio de reduccion de dimensionalidad

  Retorna
  -------
    pd.DataFrame

  """
  # Parametros de la funcion
  num_data = data.select_dtypes(["float64","int64"])
  cat_data = data.select_dtypes(["object","bool"])
  columns = list(data.columns)
  num_cols = list(data.select_dtypes(["float64","int64"]).columns)
  cat_cols = list(data.select_dtypes(["object","bool"]).columns)

  # Manejo de errores en los key word arguments
  if encodings == []:
    print("ERROR: key word argument encodings it's empty and minimum one encoder must be defined")
    return None

  # Inicializacion del df informativo
  df_columns = ['ENCODER','ORIGINAL_FEATURES','NEW_FEATURES','NUMERICAL_TRANSFORMER','PCA_COMPONENTS','ORIGINAL_VARIANCE_RETAINED']
  if num_transformers != []:

    combinations = list(product(set(encodings), set(num_transformers), set(pca_components))) # lista de tuplas de todas posibles combinaciones no repetidas: [(ecoder,transformer,n_pca),(...)]
    df_rows = len(combinations)
    print('combinations',combinations)

  else:

    combinations = list(product(set(encodings), set(pca_components)))
    df_rows = len(combinations)
    print('combinations',combinations)

  df_pca = pd.DataFrame(columns = df_columns, index = range(df_rows))


  # Manejo de de valores faltantes, PCA no maneja NA values es necesario dropearlos del dataframe
  col_with_NA = [(col,data[f"{col}"].isnull().sum()) for col in data.columns if data[f"{col}"].isnull().sum() > 0] # Lista con tuplas = (columna, numero de NA)
  print("col_with_NA: ",col_with_NA)

  # Drop de esos NA, el usuario por pantalla elige si dropear la columna entera o las filas. [si hay muchos NA en esa columna drop de columna si no de fila]
  data_drop = data.copy()
  for col_name,num_NA in col_with_NA:

    print(f"En la columna '{col_name}' con {num_NA} valores NA, hay: {data[f'{col_name}'].shape[0]} filas y el {(num_NA/data[f'{col_name}'].shape[0]) *100} % son valores NA")
    n = int(input(f"insertar: '1' para borrar la columna {col_name} o '0' para borrar sus filas con valores NA -- "))
    print("-----------------------------------------------------------------")
    while n != 1 and n != 0:
      n = input("Input error: no existe esa opcion, vuelva a introducir '1' o '0' : ")
      print("-----------------------------------------------------------------")
    if n == 1:
      data_drop.drop(labels = f'{col_name}', axis = 1, inplace = True)
    if n == 0:
      data_drop.dropna( subset = [f"{col_name}"], inplace = True)
      print(f"Actualizacion del dataframe tras el drop -- numero filas :  {data_drop[f'{col_name}'].shape[0]} ")
      print("-----------------------------------------------------------------")
    print(f"Actualizacion del dataframe tras el drop -- numero columnas :  {len(data_drop.columns)} ")
    print("-----------------------------------------------------------------")


  # Inicio de bucles
  for comb_index , combination in enumerate(combinations):

    if len(combination) == 3: # == num_transformers != []:

      # Manejo de errores en los key word arguments
      if combination[2] <= 0:

        print("ERROR: The number of PCA components must be > 0")


      else:

        # Pca object
        pca = PCA(n_components = combination[2])


        # Df fill
        df_pca.loc[comb_index, 'PCA_COMPONENTS'] = combination[2]

        # Df fill
        df_pca.loc[comb_index, 'NUMERICAL_TRANSFORMER'] = combination[1]

        # Instancia de la clase ColumnTransformer y definicion de su encoder
        preprocessor = ColumnTransformer( transformers=[

                                                        ('encoder', combination[0], list(data_drop.select_dtypes(["object","bool"]).columns)),
                                                        ],
                                          remainder='passthrough' # IMPORTANTE: columnas del df no "procesadas" en el ColumnTransformer las conserva;
                                                                  # con "drop" las dropea del df que le pasa al siguiente step del pipeline
                                        )

        # Df fill
        df_pca.loc[comb_index, 'ENCODER'] = combination[0]

        # Instancia de la clase Pipeline y definicion de sus steps (entre ellos el numerical transformer si tiene)
        pipeline = Pipeline(steps=[
                                ('preprocessor', preprocessor),
                                ('num_transf', combination[1]),
                                ("pca",pca),
                                ])
    elif len(combination) == 2: # num_transformers == []:

      # Manejo de errores en los key word arguments
      if combination[1] <= 0:

        print("ERROR: The number of PCA components must be > 0")


      else:

        # Pca object
        pca = PCA(n_components = combination[1])

        # Df fill
        df_pca.loc[comb_index, 'PCA_COMPONENTS'] = combination[1]

        # Instancia de la clase ColumnTransformer y definicion de su encoder
        preprocessor = ColumnTransformer( transformers=[

                                                        ('encoder', combination[0], list(data_drop.select_dtypes(["object","bool"]).columns)),
                                                        ],
                                          remainder='passthrough' # IMPORTANTE: columnas del df no "procesadas" en el ColumnTransformer las conserva;
                                                                  # con "drop" las dropea del df que le pasa al siguiente step del pipeline
                                        )
        # Df fill
        df_pca.loc[comb_index, 'ENCODER'] = combination[0]

        # Instancia de la clase Pipeline y definicion de sus steps (entre ellos el numerical transformer si tiene)
        pipeline = Pipeline(steps=[
                                ('preprocessor', preprocessor),
                                ("pca",pca),
                                ])

    data_transformed = pipeline.fit_transform(data_drop)
    variance = pca.explained_variance_ratio_

    codifier_name = pipeline.named_steps['preprocessor'].transformers[0][1]
    print(f'Number of PCA components choosen after using {codifier_name}: {(len(variance))}') # Si pone :n_components=None te calcula n_components == n_features
                                                                                        # De esta forma se ven la features totales credas tras one hot encoder
    # Df fill
    df_pca.loc[comb_index, 'ORIGINAL_VARIANCE_RETAINED'] = np.sum(variance)

    print(f'Explained variance ratio: \n {(variance)}') # autovalor (de cada PCA componnet) /suma de todos autovalores --- returns array numpy: dimensions == PCA component choose
    print(f'Fraction of original variance (or information) kept by each principal component axis (or image vector) after apllying {codifier_name}:{(np.sum(variance))}') # image vector == vector proyectado
    print("-----------------------------------------------------------------")

    # Plot del porcentaje de varianza retenida por cada PCA creada
    plt.figure(figsize=(12, 9), layout ='constrained',linewidth = 0.1)
    plt.bar(range(1,len(variance) +1 ), variance, alpha=1, align='center', label=f'Individual explained variance',color = 'cyan', edgecolor = 'black')
    plt.ylabel('Explained variance ratio')
    plt.xlabel(f'Principal components using {codifier_name}')
    plt.xticks(range( 1,len(variance) +1))
    plt.grid()
    plt.legend(loc='best') # Matplotlib intentará elegir la ubicación óptima de la leyenda para evitar que se solape con los datos trazados
    plt.show()


    if len(combination) == 3:
      # Plot del peso de cada feature (tras preprocesamiento, antes de PCA) en el PCA 1:
      feature_names_after_step = pipeline['preprocessor'].get_feature_names_out(input_features = data_drop.columns) #obtener nombre de las features tras un preprocesamiento en cierto step del pipeline
      feature_names_after_step_2 = pipeline['num_transf'].get_feature_names_out(input_features = feature_names_after_step) #obtener nombre de las features en el step 2 del pipeline
      print(f"Numero de features tras step {str(pipeline['preprocessor'])}: ",len(feature_names_after_step))
      print(f"Numero de features tras step {str(pipeline['num_transf'])}: ",len(feature_names_after_step_2))
      print("-----------------------------------------------------------------")

      # Df fill
      df_pca.loc[comb_index, 'ORIGINAL_FEATURES'] = len(data_drop.columns)
      df_pca.loc[comb_index, 'NEW_FEATURES'] = len(feature_names_after_step_2)

      first_principal_component = pipeline["pca"].components_[0]

      # Ploteann solo features que tenga un cierto peso/relevancia (poostivo o negativo) sobre PCA 1
      umbral = 0.1
      feature_namesfor_plotting = [i  for i,j in (zip(feature_names_after_step_2 , first_principal_component)) if abs(j) > umbral]
      first_principal_component_plotting = [j  for j in first_principal_component if abs(j) > umbral]

      plt.figure(figsize=(15, 10),layout= 'constrained')
      plt.bar(feature_namesfor_plotting, first_principal_component_plotting, color='blue', edgecolor = 'black')
      plt.xlabel('Features')
      plt.ylabel('Value')
      plt.xticks(rotation=40)
      plt.title(f'First Principal Component Weights for Each Feature using {codifier_name}')
      plt.grid()
      plt.show()

    if len(combination) == 2:

      # Plot del peso de cada feature (tras preprocesamiento, antes de PCA) en el PCA 1:
      feature_names_after_step = pipeline['preprocessor'].get_feature_names_out(input_features = data_drop.columns) #obtener nombre de las features tras un preprocesamiento en cierto step del pipeline
      print(f"Numero de features tras step {str(pipeline['preprocessor'])}: ",len(feature_names_after_step))

      # Df fill
      df_pca.loc[comb_index, 'ORIGINAL_FEATURES'] = len(data_drop.columns)
      df_pca.loc[comb_index, 'NEW_FEATURES'] = len(feature_names_after_step)

      first_principal_component = pipeline["pca"].components_[0]

      # Ploteann solo features que tenga un cierto peso/relevancia (poostivo o negativo) sobre PCA 1
      umbral = 0.1
      feature_namesfor_plotting = [i  for i,j in (zip(feature_names_after_step , first_principal_component)) if abs(j) > umbral]
      first_principal_component_plotting = [j  for j in first_principal_component if abs(j) > umbral]

      plt.figure(figsize=(15, 10), layout= 'constrained')
      plt.bar(feature_namesfor_plotting, first_principal_component_plotting, color='blue', edgecolor = 'black')
      plt.xlabel('Features')
      plt.ylabel('Value')
      plt.xticks(rotation=40)
      plt.title(f'First Principal Component Weights for Each Feature using {codifier_name}')
      plt.grid()
      plt.show()


  return df_pca

"""
# EJEMPLO DE USO DE LA FUNCION:

# Encoders (transformer objects):

onehot = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
ordinal = OrdinalEncoder()

# Transformers (transformer objects):

estandar = StandardScaler() # Estandarizacion
norm = MinMaxScaler(feature_range=(0, 1)) # Normalizacion
max_abs = MaxAbsScaler()
robust = RobustScaler(quantile_range=(25.0,75.0))

### llamada a la funcion:
df_informativo_pca = pca(data = data, encodings = [onehot], num_transformers = [estandar,norm,max_abs,robust], pca_components =  [20,1250])

"""

def seq_feature_selector(
                          X : np.ndarray,
                          y: np.ndarray,
                          estimator : object,
                          n_features_to_select : int = 1,
                          tol : Optional[float] = None,
                          direction : str = 'backward',
  
                            ) -> Tuple[np.ndarray,np.ndarray]:
  """Seq feture selector"""
  # Sklearn class import
  from sklearn.feature_selection import SequentialFeatureSelector
  
  # Error in the parameters
  if direction not in ("forward", 'backward'):
      raise ValueError(f"direction must be 'forward' or 'backward'; got {direction}")
    
  if type(n_features_to_select) != int and n_features_to_select != "auto":
      raise ValueError(f"n_features_to_select must be 'auto' or int; got {n_features_to_select}")
    
  elif type(n_features_to_select) != int and n_features_to_select == "auto" and  tol ==None:
    raise ValueError(f"tol must be defined as a criteria to select number the number of features, must be a float")
    
  # Fit object of the instance class on X,y train sets
  sfs = SequentialFeatureSelector(
                              estimator = estimator,  
                              n_features_to_select=n_features_to_select , 
                              direction = direction
                              )
  sfs.fit(X,y)
  return sfs.transform(X)
  
