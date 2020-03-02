#!/Users/akagawaoozora/opt/anaconda3/bin/python

import sys
sys.path.append("../")
from DecisionTree.DecisionTree import DecisionTree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd


TEST_SIZE = 0.5


def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    train, test = train_test_split(df, test_size=TEST_SIZE)
    return train, test


def main():
    train, test  = load_data()
    train_X, train_y = train.iloc[:, :-1], train["target"]
    test_X, test_y = test.iloc[:, :-1], test["target"]

    dt_wrap = DecisionTree(train_X, train_y, test_X, test_y) 
    dt_wrap.set_classifier(method="Decision Tree")
    dt_wrap.fit()
    dt_wrap.predict()
    dt_wrap.check_importance()


if __name__ == "__main__":
    main()
