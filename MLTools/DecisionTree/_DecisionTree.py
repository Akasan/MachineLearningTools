import warnings
warnings.simplefilter('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from .src.ConfigHandler import load_config, save_config
from .src.Visualizer import visualize_tree
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.externals.six import StringIO
from graphviz import Digraph
import xgboost as xgb
import lightgbm as lgb
import pydotplus

# import config


class DecisionTree:
    """ Wrapper of Decision Tree methods in sklearn, XGBoost and LightGBM

    Attributes:
    ----------
        __METHOD {dict} -- keys are method name and value are function
        __METRICS {dict} -- keys are metric name and value are function
        __DEFAULT_CONFIG {dict} -- keys are method name and value are the file name of each method's default parameter
        METHOD {str} -- method name you'll use for analyzing
        train_X {pd.DataFrame} -- input dataset for training
        train_y {pd.DataFrame} -- output dataset for training
        test_X {pd.DataFrame} -- input dataset for testing
        test_y {pd.DataFrame} -- output dataset for testing
        feature_name {list[str]} -- feature names
        clf {any} -- classifier instance

    Examples:
    -------
        >>> dt = DecisionTree()                                     # make wrapper which use Random Forest for analyzing
        >>> dt.METHOD                                               # confirm method
        Random Forest
        >>> dt = DecisionTree(method="Decision Tree")               # make wrapper which use Decision Tree for analyzing
        >>> dt.METHOD
        Decision Tree
        >>> dt.set_dataset(train_X, train_y, test_X, test_y)        # set datasets
        >>> dt.set_parameter(**params)                            # make classifier instance
    """

    __METHOD = {"Random Forest": RandomForestClassifier, "Decision Tree": DecisionTreeClassifier, "XGBoost": xgb.XGBClassifier}
    __METRICS = {"accuracy": accuracy_score, "roc_curve": roc_curve, "auc": auc}
    __DEFAULT_CONFIG = {"Random Forest": "../config/RandomForestConfig.json", "Decision Tree": "../config/DecisionTreeConfig.json"}

    def __init__(self, method="Random Forest"):
        """
        Keyword Arguments:
        ------------------
            method {str} -- method name you want to use for analyzing dataset (default: Random Forest)
        """
        self.METHOD = method

    def set_dataset(self, train_X, train_y, test_X, test_y):
        """ set dataset

        Arguments:
        ----------
            train_X {pd.DataFrame} -- input dataset for training
            train_y {pd.DataFrame} -- output dataset for training
            test_X {pd.DataFrame} -- input dataset for testing
            test_y {pd.DataFrame} -- output dataset for testing

        Examples:
        ---------
            >>> dt = DecisionTree()
            >>> dt.set_dataset(train_X, train_y, test_X, test_y)
        """
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.feature_name = self.train_X.columns

    def set_parameter(self, is_from_file=False, filename=None, **kwargs):
        """ make classifier instance.
        Please refer arguments by calling help_method function

        Keyword Arguments:
        ------------------
            is_from_file {bool} -- if you want to load parameters from file , set this as True
            filename {str} -- file name of parameters (default: None)

        Examples:
        --------
            >>> dt = DecisionTree()
            >>> param = {"n_estimators": 100}
            >>> dt.set_parameter(**param)
            >>> dt.clf              # show description of classifier
            RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                        max_depth=None, max_features='auto', max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                        oob_score=False, random_state=None, verbose=0,
                        warm_start=False)
            >>> dt.set_parameter(n_estimators=100)            # This is another form which is the same implementation of above
            >>> dt.set_parameter(is_from_file=True, filename="../config/RandomForestConfig.json")       # load config file
        """
        if is_from_file:
            param = load_config(filename)
            self.clf = self.__METHOD[self.METHOD](**param)
        else:
            self.clf = self.__METHOD[self.METHOD](**kwargs)

    def help_method(self):
        """ print help information of method you specified

        Examples:
            >>> dt = DecisionTree()
            >>> dt.help_method()
            The help information of Random Forest will be printed
        """
        print(help(self.__METHOD[self.METHOD]))

    # TODO 通常の方法で実装しているので、CVなども取り入れる
    def fit(self, is_show_result=True, **kwargs):
        """ fit classifier

        Arguments:
        ----------
            X {pd.DataFrame} -- dataframe of input
            y {pd.DataFrame} -- dataframe of output

        Keyword Arguments:
        ------------------
            is_show_result {bool} --whether you want to show result at the same time (default: True)

        Examples:
        ---------
            >>> dt = DecisionTreeWeap()
            >>> dt.set_parameter()
            >>> dt_wrap.fit(X, y)
        """
        self.clf.fit(self.train_X, self.train_y, **kwargs)
        accuracy = accuracy_score(self.clf.predict(self.train_X), self.train_y)

        if is_show_result:
            print(f"- Training Accuracy : {accuracy: 0.3f}")

    def predict(self, is_show_result=True):
        """ predict output. input will be selected from test_X

        Keyword Arguments:
            is_show_result {bool} -- print predict accuracy (default: True)

        Examples:
            >>> dt_wrap = DecisionTreeWrap(...)
            >>> dt_wrap.set_classifier(method="Random Forest")
            >>> dt_wrap.fit()
            >>> dt_wrap.predict()
            Accuracy : 0.912            # For example
        """
        self.pred = self.clf.predict(self.test_X)
        self.accuracy = accuracy_score(self.pred, self.test_y)

        if is_show_result:
            print(f"- Predicted Accuracy : {self.accuracy: 0.3f}")

    def check_depth(self, depth_range, **kwargs):
        """ check depth of model

        Arguments:
            depth_range
        """
        result = []

        for d in depth_range:
            self.set_classifier(max_depth=d, **kwargs)
            self.fit(self.train_X, self.train_y)
            self.predict(self.test_X, self.test_y)
            result.append(self.accuracy)

        plt.plot(depth_range, result)
        plt.show()

    def visualize_tree(self, depth=-1):
        """ visualize classifier tree

        Keyword Arguments:
            depth {int} -- how many depth you want to visualize (deafult: -1(means all tree))
        """
        visualize_tree(self.clf, self.feature_name, self.METHOD, depth)

    def check_importance(self, is_plot=False):
        """ print feature importanes

        Keyword Arguments:
            is_plot {bool} -- whether plot or not (default: False)  -> not implemented

        Examples:
            >>> dt_wrap = DecisionTreeWrap(...)
            >>> dt_wrap.set_classifier(method="Random Forest")
            >>> dt_wrap.fit()
            >>> dt_wrap.checkimportance()
            feature : A     importance : 0.3
            feature : B     importance : 0.5
            feature : C     importance : 0.2
        """
        self.importances = {k: v for k, v in zip(self.features, self.clf.feature_importances_)}

        print("========== Feature Importance ==========")
        for k, v in sorted(self.importances.items(), key=lambda x: -x[1]):
            print(f"\tFeature : {k} \tImportance : {v}")