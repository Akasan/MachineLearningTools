from graphviz import Digraph
import pydotplus
from sklearn.externals.six import StringIO
from sklearn import tree
from ._SaveGraph import write_file
from xgboost import plot_tree
import matplotlib.pyplot as plt


def visualize_tree(clf, feature_name, method, depth=-1):
    if depth == -1:
        depth = clf.max_depth
    else:
        assert 1 <= depth <= clf.tree_.max_depth, f"Please set depth from 1 to {clf.max_depth}"

    if method == "Random Forest":
        dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_name, max_depth=depth)
        graph = pydotplus.graph_from_dot_data(dot_data)
        write_file(graph, "graph", mode="pdf")

    elif method == "XGBoost":
        plot_tree(clf)
        plt.show()

    else:
        print(dir(clf))
        for i, _clf in enumerate(clf.estimators_):
            dot_data = tree.export_graphviz(_clf, out_file=None, feature_names=feature_name, max_depth=depth)
            graph = pydotplus.graph_from_dot_data(dot_data)
            write_file(graph, f"graph{i}.pdf")