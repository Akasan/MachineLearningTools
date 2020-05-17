from sklearn.datasets import *
from sklearn.model_selection import train_test_split
import pandas as pd


__PRESET_LIST = {
    "iris": load_iris,
    "boston": load_boston,
    "breast_cancer": load_breast_cancer,
    "diabetes": load_diabetes,
    "digits": load_digits,
    "files": load_files,
    "linnerud": load_linnerud,
    "wine": load_wine
}


def print_dataset_list():
    """ print dataset list available to load from sklearn.datasets

    Examples:
    ---------
        >>> print_dataset_list()
        - iris
        - boston
        - breast_cancer
        - diabetes
        - digits
        - files
        - linnerud
        - wine
    """
    for k in __PRESET_LIST.keys():
        print(f"- {k}")


def _to_df(data):
    """ convert from sklearn.utils.Bunch to pd.DataFrame

    Arguments:
    ----------
        data {sklearn.utils.Bunch} -- original data from sklearn.dataset

    Returns:
    --------
        {pd.DataFrame} -- pd.DataFrame format

    Examples:
    ---------
        >>> iris = sklearn.datasets.loadd_iris()
        >>> iris_df = _to_df(iris)
        >>> type(iris_df)
        <class 'pd.DataFrame'>
    """
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target_names[data.target]
    return df


def load_dataset(name, is_separate=False, is_split_dataset=False, train_split_rate=0.7):
    """ load dataset

    Arguments:
    ----------
        name {str} -- dataset name
                      Please check available name list by calling print_dataset_list

    Keyword Arguments:
    ------------------
        is_separate {bool} -- whether separate data into input values and output values (default: False)
        is_split_dataset {bool} -- whether split dataset into for training and testing (default: False)
        train_split_rate {float} -- split rate for training (default: 0.7)

    Examples:
    ---------
        >>> iris_df = load_dataset("iris")                               # get dataset, without separating X and y
        >>> iris_X, iris_y = load_dataset("iris", is_separate=True)      # get dataset
    """
    if not name in __PRESET_LIST:
        raise Exception(f"{name} dataset is invalid.")

    data = __PRESET_LIST[name]()
    df = _to_df(data)

    if is_split_dataset:
        train, test = train_test_split(df, train_size=train_split_rate)

        if is_separate:
            return train.drop("target", axis=1), train.target, test.drop("target", axis=1), test.target

        else:
            return train, test

    else:
        if is_separate:
            return df.drop("target", axis=1), df.target

        else:
            return df