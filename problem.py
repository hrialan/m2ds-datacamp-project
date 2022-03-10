# https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/problem.html#problem
# https://paris-saclay-cds.github.io/ramp-workflow/workflow.html#

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

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

_target_column_name = 'price'


def _read_data(path=".", split="train"):
    df_main = pd.read_csv(path + 'airbnb_paris.csv', index_col='id')
    df_listings = pd.read_csv(path + 'listings.csv', index_col='id')

    listings_cols_to_keep = ["host_is_superhost",
                             "accommodates",
                             "bedrooms",
                             "beds",
                             "availability_60",
                             "number_of_reviews_l30d",
                             "availability_90",
                             "review_scores_accuracy",
                             "review_scores_cleanliness",
                             "review_scores_checkin",
                             "review_scores_communication",
                             "review_scores_location",
                             "review_scores_value"]

    df = pd.merge(df_main,
                  df_listings[listings_cols_to_keep], on='id', how='left')

    del df['neighbourhood_group']
    del df['license']
    del df['last_review']
    df['host_name'] = df['host_name'].fillna("NO_NAME")

    def process_empty(x):
        x = x.replace(' ', '')
        return x if x != '' else np.NaN

    for col in df.columns:
        try:
            df[col] = df[col].apply(process_empty)
        except AttributeError:
            pass

    data_type = {
        'int': ['id', 'price', 'host_id', 'minimum_nights',
                'calculated_host_listings_count',
                'availability_365', 'number_of_reviews_ltm'],
        'float': ['latitude', 'longitude', 'reviews_per_month'],
        'datetime': ['last_review'],
        'object': ['name', 'host_name', 'room_type']
    }

    for col in df.columns:
        for key, values in data_type.items():

            if col in values:
                if key == 'datetime':
                    df[col] = pd.to_datetime(df[col])
                    df['last_review'] \
                        = df['last_review'].dt.strftime('%d-%m-%Y')
                else:
                    df[col] = df[col].astype(key)

    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    df['host_is_superhost'] = df['host_is_superhost'].fillna('f')
    df['bedrooms'] = df['bedrooms'].fillna(0)
    df['beds'] = df['beds'].fillna(1)
    df['name'] = df['name'].fillna(" ")

    scores = ['accuracy', 'cleanliness',
              'checkin', 'communication',
              'location', 'value']
    df['review_scores_' + scores[0]].isna().sum() / df.shape[0]

    for elt in scores:
        index_with_nan = df.index[df[['review_scores_' +
                                  elt]].isnull().any(axis=1)]
        df.drop(index_with_nan, axis=0, inplace=True)

    df = df[df[_target_column_name] > 0]

    filtered_entries = []
    for room_type in df['room_type'].unique():
        condition = df['room_type'] == room_type
        df_room_filtered = df[condition]
        z_scores = stats.zscore(df_room_filtered[_target_column_name])
        abs_z_scores = np.abs(z_scores)
        # filter the values less than 3 standard deviations from the mean
        filtre = abs_z_scores < 3
        filtered_entries.append(filtre)

    rows_to_keep = pd.concat(filtered_entries)
    rows_to_keep.sort_index(inplace=True)
    df = df[rows_to_keep]

    df_model = df.drop(columns=['name', 'host_id', 'host_name'])

    X = df_model.drop(['price'], axis=1)
    y = df_model['price']

    for object_col in X.columns[X.dtypes == 'object']:
        enc = OrdinalEncoder()
        enc.fit(X[[object_col]])
        enc.categories_
        X[object_col] = enc.transform(X[[object_col]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=X["room_type"].values.tolist())

    if split == "train":
        return X_train.to_numpy(), y_train.to_numpy()
    if split == "test":
        return X_test.to_numpy(), y_test.to_numpy()


def get_train_data(path=".", split='train'):
    return _read_data(path="./data/", split=split)


def get_test_data(path=".", split='test'):
    return _read_data(path="./data/", split=split)


def get_cv(X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    return cv.split(X, y)
