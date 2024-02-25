from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from typing import Tuple,List,Optional
import numpy as np
from sklearn.preprocessing import LabelEncoder

def synthetic_resample(
                            X : np.ndarray,
                            y : np.ndarray,
                            ratio : float ,
                            technique : str = "SMOTE",
                            verbose : int = 0
                        ) -> Tuple[np.ndarray,np.ndarray]:
    """_summary_

    Args:
        X (np.ndarray): 
        y (np.ndarray): y array should be labely codify previously : ["class0","class1",...] -> [0,1,2,...]
        ratio (float): ratio of resampling compared with more frequent class
        technique (str): technique for resampling : ["SMOTE", "oversampling","undersampling"]
        verbose (int, optional): 1 fore extra information. Defaults to 0. 

    Returns:
        Tuple[np.ndarray,np.ndarray]: --> x_resample, y_resample
    """
    if X.__class__ != 'numpy.ndarray' or y.__class__ != 'numpy.ndarray':
        raise TypeError("X and/or y should be numpy arrays")
    
    print(f"y type : {y.dtype}")
    if y.dtype != 'int64' or y.dtype != 'int64':
        print("TypeError : y should be codify previously, a label encoder will be applied")
        y = LabelEncoder().fit_transform(y)
        print(f"y new type : {y.dtype}")
        
    if verbose == 1:
        print(f"Original dataset number of samples : {X.shape[0] :.2f} ")
        print("Classes : ", np.unique(y))
        print(f"Class frequencies : {np.bincount(y)}")
        for idx, v in enumerate(np.bincount(y)):
            print(f'Proportion of class {idx} : {100*v/y.shape[0] : .3f}','%')

        
    min_class = np.argmin(np.bincount(y))
    max_class = np.argmax(np.bincount(y))
    new_class_samples = int(ratio*(np.bincount(y)[max_class]))
    
    print("-----------------------------------------------------------")
    print("Minority class : ", min_class)
    print("Mayority class : ", max_class)
    print("New number of samples : ",new_class_samples)
    print("-----------------------------------------------------------")
    
    if verbose == 1:
        print(X_resample.shape, y_resample.shape)
        print(f"% of increment compare to original dataset : {100*(X_train_oversampled.shape[0]-X_train.shape[0])/X_train.shape[0]:.2f} %")
        print("New codify classes : ", np.unique(y_resample))
        print(f"Class frequencies : {np.bincount(y_resample)}")
        for idx, v in enumerate(np.bincount(y_resample)):
            print(f'Proportion of class {idx} : {100*v/y_resample.shape[0] : .3f}','%')
    


def testing() -> None:
    x = np.array([1,2,3,4,5,6,7.,8])
    x = np.array(["a","b","c","d","e",1,1,3.])
    x = np.array(["a","b","c","d","e"])
    
    x = np.array([[1,2,3,7,8],["a","b","c","d","e"]])
    
    print(x.dtype)
    print(x.__class__)
    
    
if __name__ == "__main__":
    testing()