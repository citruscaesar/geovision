from numpy.typing import NDArray
import numpy as np
import pandas as pd

def calculate_confusion_matrix(logits: NDArray) -> NDArray:
    return np.eye(10)

def calculate_classification_metrics(confusion_matrix: NDArray) -> pd.DataFrame:
    return pd.DataFrame()
