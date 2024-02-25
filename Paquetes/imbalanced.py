from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from typing import Tuple,List,Optional
import numpy as np

def synthetic_resample(
                            X : np.ndarray,
                            y : np.ndarray,
                            ratio : float ,
                            verbose : int = 0
                        ) -> Tuple[np.ndarray,np.ndarray]:
    """_summary_

    Args:
        X (np.ndarray): 
        y (np.ndarray): y array should be labely codify previously : ["class0","class1",...] -> [0,1,2,...]
        ratio (float): ratio of resampling compared with more frequent class
        verbose (int, optional): 1 fore extra information. Defaults to 0. 

    Returns:
        Tuple[np.ndarray,np.ndarray]: --> x_resample, y_resample
    """
    if X.__class__ != 'numpy.ndarray' or y.__class__ != 'numpy.ndarray':
        raise TypeError("X and/or y should be numpy arrays")
    
    if y.dtype != 'int64' or y.dtype!= 'int64':
        pass
    min_class = np.argmin(np.bincount(y))
    max_class = np.argmax(np.bincount(y))
    new_minority_class_samples = int(ratio*(np.bincount(y)[max_class]))
    new_minority_class_samples = int(ratio*(np.bincount(y)[max_class]))
    

    if verbose == 1:
        pass


def testing() -> None:
    x = np.array([1,2,3,4,5,6,7.,8])
    x = np.array(["a","b","c","d","e",1,1,3.])
    x = np.array(["a","b","c","d","e"])
    
    x = np.array([[1,2,3,7,8],["a","b","c","d","e"]])
    
    print(x.dtype)
    print(x.__class__)
    
    
if __name__ == "__main__":
    testing()