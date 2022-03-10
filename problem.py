# https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/problem.html#problem

import os

import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

import rampwf as rw
from rampwf.score_types.base import BaseScoreType

problem_title = "Airbnb price prediction"


Predictions = rw.prediction_types.make_regression()
workflow = rw.workflows.Regressor()

class MAPE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, name="MAPE", precision=4):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        mape = (np.abs(y_true - y_pred) / y_true).mean()
        return mape


score_types = [
    MAPE(name="MAPE"),
]

def get_cv(X, y):
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=1)
    return cv.split(X, y)

_target_column_name = 'price'
_ignore_column_names = []

def _read_data(path, f_name):
    # TODO 
    return 0


def get_train_data(path='.'):
     # TODO 
    return 0


def get_test_data(path='.'):
     # TODO 
    return 0