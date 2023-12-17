import os
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

import rampwf as rw

problem_title = "Hot Jupiter atmospheric pattern classification"


_event_label_names = [
    "asymetric",
    "banded",
    "locked",
    "butterfly",
    "no_pattern",
]

# Correspondence between categories and int8 categories
# Mapping int to categories
int_to_cat = {
    0: "asymetric",
    1: "banded",
    2: "locked",
    3: "butterfly",
    4: "no_pattern",
}
# Mapping categories to int
cat_to_int = {v: k for k, v in int_to_cat.items()}

_event_label_int = list(int_to_cat)

Predictions = rw.prediction_types.make_multiclass(label_names=_event_label_int)
workflow = rw.workflows.Classifier()


score_types = [
    rw.score_types.BalancedAccuracy(
        name="bal_acc", precision=3, adjusted=False
    ),
    rw.score_types.Accuracy(name="acc", precision=3),
]

# Create global variable to use in LOGO CV strategy
groups = None


def _get_data(path=".", split="train"):
    # Load data from npy and csv files.
    # Data: raw images .npy format
    # Labels: .csv file
    #
    # returns X (input) and y (output) arrays
    # X: array of shape (n_samples, 90, 180)
    # y: array of shape (n_samples,)

    # data
    data = np.load(os.path.join(path, "data", "X_" + split + ".npy"))
    X = data

    # labels
    y_df = pd.read_csv(os.path.join(path, "data", "y_" + split + ".csv"))
    y = y_df.cat_num.to_numpy()

    return X, y


def get_train_data(path="."):
    # Load y_df from file
    y_df = pd.read_csv(os.path.join(path, "data/y_train.csv"))
    # Create meta groups as a new column in y_df
    # Simulations to groups dictionary
    sim_to_group = {
        "Cool_0060_Locked": 0,
        "Cool_0110_match": 0,
        "Hot_0012_match": 0,
        "Hot_0036_Locked": 0,
        "Cool_0334_Locked": 1,
        "Hot_0012_Locked": 1,
        "Cool_0060_match": 1,
        "Cool_0110_Locked": 2,
        "Cool_0192_match": 2,
        "Cool_0334_match": 2,
        "Hot_0021_Locked": 2,
        "Hot_0021_match": 2,
    }

    # Function to apply to each row in the 'simulation' column
    def get_group(simulation):
        return sim_to_group.get(simulation, None)

    # Create a new column 'group' based on the 'simulation' column
    y_df["group"] = y_df["simulation"].apply(get_group)

    # Gobal variable groups
    global groups
    groups = y_df.group.to_numpy()

    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")


def get_cv(X, y):
    cv = LeaveOneGroupOut()
    return cv.split(X, y, groups=groups)
