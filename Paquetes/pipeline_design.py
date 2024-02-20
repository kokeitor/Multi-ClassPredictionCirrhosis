import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,VarianceThreshold,f_classif
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor

### Scoring
from sklearn.model_selection import cross_validate

### Importacion custom transformers class
from  custom_transformers import CustomDropper, LabelCustomEnc
### Importacion custom transformers functions
from sklearn.preprocessing import FunctionTransformer
from  custom_transformers import combine_date_columns, reducir_categorias_columna,convertir_columna_categorica, tratamiento_de_outliers, tratamiento_datos_faltantes


### Deteccion y alamacenamiento de features segun sus tipos de datos en listas:

"""
categorical_cols = list(X_train.select_dtypes(include=['object', 'bool']).columns) # Detect de categortiacl features of dataframe
numerical_cols = list(X_train.select_dtypes(include=['int64', 'float64']).columns) # Detect all numerical features of dataframe
date_cols = list(X_train.select_dtypes(include=['datetime64[ns]']).columns)
print("categorical_cols: ",categorical_cols)
print("numerical_cols: ",numerical_cols)
print("datetime_cols: ",date_cols)

numerical_cols.remove("agent")
numerical_cols.remove("company")
"""
# categorical_cols.remove("country")
# categorical_cols.append("agent")
# categorical_cols.append("company")
# categorical_cols.append("country2")

###########################################################################################
############################## Transformer objects ########################################
###########################################################################################

### Categorical feature transformer objects::-------------------------------------------------------
# Categorical imputers:

### Feature engineering transformers -----------------------------------------------------------------
reduccion_categorias = FunctionTransformer(
                                        func = reducir_categorias_columna,
                                        validate = False,
                                         kw_args={
                                                  "columna": ["agent","company","country"],
                                                  "nuevaColumna" : ["agent","company","country"],
                                                  "cantCategorias" : 10
                                                  }
                                        )

### Column/feature dropper transformer objects::-------------------------------------------------------

dropper = CustomDropper(column_transform = date_cols)


# Categorical codifiers:
onehot = OneHotEncoder(handle_unknown='ignore',sparse_output = False)
labelEnc = LabelCustomEnc(column_transform = categorical_cols)
ordinalenc = OrdinalEncoder(handle_unknown="use_encoded_value", encoded_missing_value=100, unknown_value=100)
                                                                      # encoded_missing_value: codifica los valores NA o np.nan (es lo mismo) con el int que se quiera
                                                                      # en este caso si encuentra un NA/np.nan lo establece como un 20
                                                                      # unknown_value: en el caso de que se encuentre una categoria inesperada/nueva le asigna este entero

### Encoder de fechas -----------------------------------------------------------------------
date_combiner = FunctionTransformer(func = combine_date_columns , validate = False) # Crear el objeto transformer personalizado mediante la instancia de FunctionTransformer

### Transformar columnas numericas a categoricas ---------------------------------------------
transf_num_a_cat = FunctionTransformer(
                                          func = convertir_columna_categorica,
                                          validate = False,
                                          kw_args={"columna":["agent","company"]}
                                          )
### Tratamiento de outliers -------------------------------------------------------------------
outliers_treatment = FunctionTransformer(
                                          func = tratamiento_de_outliers,
                                          validate = False,
                                          kw_args={"columnas_numericas": numerical_cols }
                                          )



### Na imputers --------------------------------------------------------------------------------

mean_impt = SimpleImputer(missing_values=0, strategy='mean')
median_impt = SimpleImputer(missing_values=0, strategy='median')

na_imputer_numeric = SimpleImputer(missing_values= np.nan, strategy='median')
na_imputer_numeric.set_output(transform="pandas")

na_imputer_categoric = SimpleImputer(missing_values= np.nan, strategy='constant', fill_value = "No Agent")
na_imputer_categoric.set_output(transform="pandas")

na_imputer_categoric_2 =  SimpleImputer(missing_values= np.nan, strategy='constant', fill_value = "No Company")
na_imputer_categoric_2.set_output(transform="pandas")

NA_imputer = FunctionTransformer(
                                      func = tratamiento_datos_faltantes,
                                      validate = False,
                                    )


# Transforma NA en nueva categoria: "No <nombre_columna>"
transf_valoresNA = FunctionTransformer(
                                          func = tratamiento_datos_faltantes,
                                          validate = False,
                                          )

knn_impt = KNNImputer(missing_values=0, n_neighbors=2, weights='uniform', metric='nan_euclidean')


# Numerical transformers ------------------------------------------------------------------------------
std = StandardScaler() # Estandarizacion
minmax= MinMaxScaler(feature_range=(0, 1)) # Normalizacion
max_abs_scaler = MaxAbsScaler()
robust_scaler = RobustScaler(quantile_range=(25.0,75.0))

### Feature extraction objects:
# Dimensionality reduction:
pca = PCA(n_components = 400)

### Feature selection objects:
kbest= SelectKBest(score_func = f_classif ,k=7)
lowvariance =VarianceThreshold(threshold=0.0) #NO APLICA, NO HAY FEATURES CON LOW VARIANCE

##########################################################################################
############################## Estimators objects ########################################
##########################################################################################

rf_regressor = RandomForestRegressor(n_estimators = 5 ,bootstrap = True, random_state = 1)
sgd = SGDRegressor()


##########################################################################################
############################## Column Transformers objects ################################
##########################################################################################

## Numerical imputers: : -----------------------------------------------------------------

imputer  = ColumnTransformer(
                                    transformers=[
                                                     ('Numeric NA imputer', na_imputer_numeric , ["children"]),
                                                     ('cat 1 imputer', na_imputer_categoric, ["agent"]),
                                                     ('cat 2 imputer', na_imputer_categoric_2, ["company"]),
                                                    ],
                                    remainder='passthrough'
                                 )

## Categorical codifiers: : --------------------------------------------------------------
Encoder = ColumnTransformer(
                                    transformers=[
                                                    ('Categorical codifier', onehot, (categorical_cols)), # NOTA: PROBAR A INCLUIR COLUMNA NUMERICAS DISCRETAS !!!
                                                    ],
                                    remainder='passthrough'
                                 )

cat_index = pd.Index(['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type', 'reservation_status_date','agent','company'])
print("cat_index" , cat_index)
cat_index.astype('str')
Encoder_2 = ColumnTransformer(
                                    transformers=[
                                                    ('Categorical codifier', onehot, (cat_index)), # NOTA: PROBAR A INCLUIR COLUMNA NUMERICAS DISCRETAS !!!
                                                    ],
                                    remainder='passthrough'
                                 )


### DOCUMENTACION:

### CLASS: ColumnTransformer
# Applies transformers to columns of an array or pandas DataFrame. [apica diferentes transformadores a diferentes columnas de un mismo df(por ejemplo solo a columnas categoricas)]
# This estimator allows different columns or column subsets of the input to be transformed separately and the features generated
# by each transformer will be concatenated to form a single feature space. This is useful for heterogeneous or columnar data, to
# combine several feature extraction mechanisms or transformations into a single transformer.

# Instancia la clase ColumnTransformer creando el objeto "preprocessor" y se define uno de los atributos del objeto creado: los transformers [existen mas atributos dentro de la clase, mirar documentacion]
# el atributo transformer se define como una ¡lista de tuplas! con el siguiente formato:
# ("nombre que se quiera dar al transf", nombre del objeto Transformador(ejemplo:categorical_transformer),lista con datos tipo str que son las label (nombres) de las columnas del df q se quiere transformar )
preprocessor = ColumnTransformer(
                                  transformers=[("num",num_transformer,numerical_cols)],
                                   remainder='passthrough' #IMPORTANTE: columnas del df no "procesadas" en el ColumnTransformer las conserva a traves del pipeline
                                                            # con "drop" las dropea del df que le pasa al siguiente step del pipeline
                                 )


### CLASS: Pipeline
# Transformers are usually combined with classifiers, regressors or other estimators to build a composite estimator. The most common tool to execute this task is: Pipeline
# Pipeline can be used to chain (encadenar) multiple transformers and estimators into one.
# This is useful as there is often a fixed sequence of steps in processing the data, for example feature selection, normalization and classification
# All estimators in a pipeline, except the last one, must be transformers (i.e. must have a transform method). The last estimator may be any type (transformer, classifier, etc.)

# Note: Calling fit on the pipeline is the same as calling fit on each estimator (o trasnformer [cuando habla de transformer nos refreimos a los transformers definidos como atributo del
#       objeto preprocessor, creado de instanciar la clase: ColumnTransformer])
#       in turn, transform the input and pass it on to the next step.
#       The pipeline has all the methods that the last estimator in the pipeline has, i.e. if the last estimator is a classifier, the Pipeline
#       can be used as a classifier. If the last estimator is a transformer, again, so is the pipeline.

# Creating a pipeline: Instanciar la clase Pipeline (creando un obj tipo pipeline) y para el atributo "steps" definir una ¡lista de tuplas! == (key,value)
# la key es el nombre que se le quiera dar a es step (para luego acceder a el; y el value es el nombre del objeto: transformer o estimator)

# Nota: a cada step se accede por:
# nombre del step == pipe[""]
# por posicion del step como si fuese una lista == pipe[0]




##########################################################################################
############################## Pipelines objects #########################################
##########################################################################################


pipeline_1 = Pipeline(
                                    steps=[
                                                ('transf_num_a_cat',FunctionTransformer(
                                                                                        func = convertir_columna_categorica,
                                                                                        validate = False,
                                                                                        kw_args={"columna":["agent","company"]}
                                                                                        )
                                                ),

                                                ('NA_imputer', FunctionTransformer(
                                                                                    func = tratamiento_datos_faltantes,
                                                                                    validate = False,
                                                                                  )
                                                ),
                                                ('reduccion_categorias', FunctionTransformer(
                                                                                              func = reducir_categorias_columna,
                                                                                              validate = False,
                                                                                              kw_args={
                                                                                                        "columna": ["agent","company","country"],
                                                                                                        "nuevaColumna" : ["agent","company","country"],
                                                                                                        "cantCategorias" : 10
                                                                                                      }
                                                                                            )
                                                ),
                                                ('tratamiento_de_outliers', FunctionTransformer(
                                                                                            func = tratamiento_de_outliers,
                                                                                            validate = False,
                                                                                            kw_args={"columnas_numericas": ['is_canceled', 'lead_time', 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list', 'required_car_parking_spaces', 'total_of_special_requests'] }
                                                                                            )
                                                    ),
                                                ('Encoder', ColumnTransformer(
                                                                                  transformers=[
                                                                                                  ('Categorical codifier', onehot, (categorical_cols)),
                                                                                                  ],
                                                                                  remainder='passthrough'
                                                                              )

                                                ),
                                                ("Estimator",RandomForestRegressor(
                                                                                      n_estimators = 5 ,
                                                                                      bootstrap = True,
                                                                                      random_state = 1
                                                                                   )
                                                )
                                          ]
                                     )


pipeline_1_with_pca = Pipeline(
                  steps=[

                        ('reduccion_categorias', FunctionTransformer(
                                                                      func = reducir_categorias_columna,
                                                                      validate = False,
                                                                      kw_args={
                                                                                "columna": ["country"],
                                                                                "nuevaColumna" : ["country"],
                                                                                "cantCategorias" : 10
                                                                               }
                                                                     )
                        ),
                        ('Encoder', Encoder),
                        ('std',std),
                        ('PCA',pca),
                        ("Estimator",rf_regressor)
                        ]
                 )


pipeline_1_without_pca_outliers_1 = Pipeline(
                                              steps=[
                                                    ('tratamiento_de_outliers', FunctionTransformer(
                                                                                                      func = tratamiento_de_outliers,
                                                                                                      validate = False,
                                                                                                      kw_args={"columnas_numericas": numerical_cols }
                                                                                                      )
                                                    ),
                                                    ('reduccion_categorias', FunctionTransformer(
                                                                                                  func = reducir_categorias_columna,
                                                                                                  validate = False,
                                                                                                  kw_args={
                                                                                                            "columna": ["country"],
                                                                                                            "nuevaColumna" : ["country"],
                                                                                                            "cantCategorias" : 10
                                                                                                          }
                                                                                                )
                                                    ),
                                                    ('Encoder', Encoder),
                                                    ("Estimator",rf_regressor)
                                                    ]
                                            )

pipeline_1_without_pca_outliers_2 = Pipeline(
                                              steps=[
                                                    ('tratamiento_de_outliers', FunctionTransformer(
                                                                                                      func = tratamiento_outliers_2,
                                                                                                      validate = False,
                                                                                                      kw_args={"columnas_numericas": numerical_cols }
                                                                                                      )
                                                    ),
                                                    ('reduccion_categorias', FunctionTransformer(
                                                                                                  func = reducir_categorias_columna,
                                                                                                  validate = False,
                                                                                                  kw_args={
                                                                                                            "columna": ["country"],
                                                                                                            "nuevaColumna" : ["country"],
                                                                                                            "cantCategorias" : 10
                                                                                                          }
                                                                                                )
                                                    ),
                                                    ('Encoder', Encoder),
                                                    ("Estimator",rf_regressor)
                                                    ]
                                                    )

