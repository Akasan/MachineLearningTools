import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import sklearn
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.externals.six import StringIO
from graphviz import Digraph
import pydotplus


class DecisionTreeWrap:
    __METHOD = {"Random Forest": RanfomForestClassifier}
    
    def __init__(self, train_X, train_y, test_X, test_y):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.features = train_X.columns

    def set_classifier(self, method="Random Forest", **kwargs):
        self.clf = self.__METHOD[method](**kwargs)

    def fit(selfm **kwargs):
        self.clf.fit(self.train_X, self.train_y, **kwargs)

    def predict(self, is_show_result=True):
        self.pred = self.clf.predict(self.test_X)
        self.accuracy = accuracy_score(self.pred, self.test_y)

        if is_show_result:
            print(f"Accuracy : {self.accuracy: 0.3f}")

    def check_depth(self, depth_range, **kwargs):
        result = []

        for d in depth_range:
            self.set_classifier(max_depth=d, **kwargs)
            self.fit(self.train_X, self.train_y)
            self.predict(self.test_X, self.test_y)
            result.append(self.accuracy)

        plt.plot(depth_range, result)
        plt.show()

    def visualize_tree(self, depth=-1):
        if depth == -1:
            depth = self.clf.get_depth()
        else:
            assert 0 <= depth <= self.clf.get_depth(), f"Please set depth from 1 to {self.clf.get_depth'(}"
        
        dot_data = StringIO()
        tree.export_graphviz(clf, out_file=dot_data, feature_name=self.features, max_depth=depth)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf("graph.pdf")

    def check_importance(self, is_plot=False):
        self.importances = {k: v for k, v in zip(self.features, self.clf.feature_importances_)}

        for k, v in sorted(self.importances.items(), key=lambda x: -x[1]):
            print(f"feature : {k} \timportance : {v}")


if __name__ == "__main__":
    sample = "hoge.csv"
    df = pd.read_csv("hoge.csv")
    dt = DecisionTreeWrap(filename=sample)
