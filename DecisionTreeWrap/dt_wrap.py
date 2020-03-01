import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.externals.six import StringIO
from graphviz import Digraph
import pydotplus


class DecisionTreeWrap:
    __METHOD = {"Random Forest": RandomForestClassifier}
    
    def __init__(self, train_X, train_y, test_X, test_y):
        """
        Arguments:
            train_X {pandas.DataFrame} -- inputs for training
            train_y {pandas.DataFrame} -- outputs for training
            test_X {pandas.DataFrame} -- inputs for testing
            test_y {pandas.DataFrame} -- outputs for testing

        Examples:
            >>> dt_warap = DecisionTreeWrap(train_X, train_y, test_X, test_y))
        """
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.features = train_X.columns

    def method_list(self):
        """ print method list

        Examples:
            >>> dt_wrap = DecisionTreeWrap(...)
            >>> dt_wrap.method_list()
            Random Forest
        """
        for k in self.__METHOD.keys():
            print(k)

    def set_classifier(self, method="Random Forest", **kwargs):
        """ select classifier method 
        You can check method list by calling method_list

        Arguments:
            method {str} -- method name

        >>> dt_wrap = DecisionTreeWrap(data)
        >>> dt_wrap.set_classifier(method="Random Forest")   # use Random Forest Classifier
        """
        self.METHOD = method
        self.clf = self.__METHOD[method](**kwargs)

    @property
    def method(self):
        """ get classification method

        Examples:
            >>> dt_warap = DecisionTreeWrap(...)
            >>> dt_wrap.set_classifier(method="Random Forest", **kwargs)
            >>> dt_wrap.method
            Random Forest
        """
        return self.METHOD

    def fit(self, **kwargs):
        """ fit classifier

        Examples:
            >>> dt_wrap = DecisionTreeWeap(train_X, train_y, test_X, test_y)
            >>> dt_wrap.set_classifier(method="Random Forest", **kwargs)
            >>> dt_wrap.fit()
        """
        self.clf.fit(self.train_X, self.train_y, **kwargs)

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
            print(f"Accuracy : {self.accuracy: 0.3f}")

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
        if depth == -1:
            depth = self.clf.get_depth()
        else:
            assert 0 <= depth <= self.clf.get_depth(), f"Please set depth from 1 to {self.clf.get_depth()}"
        
        dot_data = StringIO()
        tree.export_graphviz(clf, out_file=dot_data, feature_name=self.features, max_depth=depth)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf("graph.pdf")

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

        for k, v in sorted(self.importances.items(), key=lambda x: -x[1]):
            print(f"feature : {k} \timportance : {v}")
