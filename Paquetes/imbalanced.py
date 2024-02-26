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
                            verbose : int = 0,
                            random_state : int = 1,
                        ) -> Tuple[np.ndarray,np.ndarray]:
    """_summary_

    Args:
        X (np.ndarray): 
        y (np.ndarray): y array should be labely codify previously : ["class0","class1",...] -> [0,1,2,...]
        ratio (float): ratio of resampling compared with more frequent class
        technique (str): technique for resampling : ["SMOTE", "oversampling","undersampling"]
        verbose (int, optional): 1 fore extra information. Defaults to 0. 
        random_state (int): random state. Defaults to 1.

    Returns:
        Tuple[np.ndarray,np.ndarray]: --> x_resample, y_resample
    """
    # Checking data types
    if not isinstance(X, (list,np.ndarray)) or not isinstance(y,(list,np.ndarray)):
        raise TypeError("X and/or y should be an array-like object (list or numpy array)")
    
    # Transforming list to arrays
    if isinstance(X, (list)):
        X = np.array(X)
    if isinstance(y,(list)): 
        y = np.array(y)
    
    print(f"y type : {y.dtype}",y)
    print(f"X type : {X.dtype}",X)
    old_classes = np.unique(y)
    
    if not _is_numeric(y)[0]:
        print(f"TypeError : y {_is_numeric(y)[1]} , y should be codify previously \nNote: A label encoding will be applied to y")
        y = LabelEncoder().fit_transform(y)
        if verbose == 1:
            print(f"y new type : {y.dtype}")
            print(f"y old classes : {old_classes}")
            print(f"y new codify classes : {np.unique(y)}")
        
    if not _is_numeric(X)[0]:
        raise TypeError(f"X {_is_numeric(X)[1]}")
    else:
        print(f"X {_is_numeric(X)[1]}")
        
    if verbose == 1:
        print("-----------------------------------------------------------")
        print(f"Original dataset number of samples : {X.shape[0] :.2f} ")
        print("Classes in the target variable : ", np.unique(y))
        print(f"Class frequencies : {np.bincount(y)}")
        for idx, v in enumerate(np.bincount(y)):
            print(f'Proportion of class {idx} : {100*v/y.shape[0] : .2f}','%')

        
    min_class = np.argmin(np.bincount(y))
    max_class = np.argmax(np.bincount(y))
    new_class_samples = int(ratio*(np.bincount(y)[max_class]))
    
    # OPTION RESAMPLING DICTIONARY:
    resampling_options = {
                            "SMOTE": SMOTE(
                                            sampling_strategy = ratio,
                                            random_state=random_state,
                                            ), 
                            "oversampling": None, 
                            "undersampling" :None
                        }
    # Resampling
    if (tool := resampling_options.get(technique)) !=  None:
        print(tool)
        X_resampled, y_resampled= tool.fit_resample(X, y)
    elif (tool := resampling_options.get(technique)) == None and technique == "oversampling":
        X_resampled, y_resampled = resample(
                                                X[y == min_class],
                                                y[y == min_class],
                                                replace = True,
                                                n_samples = new_class_samples,
                                                random_state=random_state
                                            )
        X_resampled = np.vstack((X[y != min_class],X_resampled))
        y_resampled = np.hstack((y[y!= min_class],y_resampled))
        
    elif (tool := resampling_options.get(technique)) == None and technique == "undersampling":
        X_resampled, y_resampled = resample(
                                                X[y == max_class],
                                                y[y == max_class],
                                                replace = False,
                                                n_samples = new_class_samples,
                                                random_state = random_state
                                            )
        X_resampled = np.vstack((X[y != max_class],X_resampled))
        y_resampled = np.hstack((y[y != max_class],y_resampled))
        
    else:
        raise ValueError(f"{technique} is not a valid resampling technique")
    
    if verbose == 1:
        print(X_resampled.shape, y_resampled.shape)
        print("-----------------------------------------------------------")
        print(f"% of increment compare to original dataset : {100*(X_resampled.shape[0]-X.shape[0])/X.shape[0]:.2f} %")
        print("New codify classes : ", np.unique(y_resampled))
        print(f"Class frequencies : {np.bincount(y_resampled)}")
        for idx, v in enumerate(np.bincount(y_resampled)):
            print(f'Proportion of class {idx} : {100*v/y_resampled.shape[0] : .2f}','%')
            
    return X_resampled, y_resampled
    

def _is_numeric(X : np.ndarray) -> Tuple[bool,str]:
    """_summary_

    Args:
        X (np.ndarray): _description_

    Returns:
        Tuple[bool,str]: _description_
    """
    # Flatten the array to handle multi-dimensional arrays
    X_flat = X.flatten()
    
    # Check if the flattened array contains integers
    if np.issubdtype(X_flat.dtype, np.integer):
        return True, "contains integer data types"
    # Check if the flattened array contains floats
    elif np.issubdtype(X_flat.dtype, np.floating):
        return True, "contains float data types"
    # Check if the flattened array contains a mix of integers and floats
    elif np.issubdtype(X_flat.dtype, np.number):
        return True, "contains a mix of integer and float data types"
    else:
        return False, "does not contain numeric data type"
        
        
        
def testing() -> None:
    x = np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]).T
    y = np.array(["a","b","b","b","a"])
    y =  ["A","B","B","B","A"]
    x_new , y_new = synthetic_resample(
                            X = x,
                            y  = y,
                            ratio = 0.7 ,
                            technique = "SMOTE",
                            verbose  = 1
                            )


if __name__ == "__main__":
    testing()