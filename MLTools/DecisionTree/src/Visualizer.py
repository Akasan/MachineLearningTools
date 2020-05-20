from graphviz import Digraph
import pydotplus
from sklearn.externals.six import StringIO
from sklearn import tree


# sklearn標準の可視化手法を使う

def _save_png(graph, filename):
    graph.write_png(filename)


def _save_pdf(graph, filename):
    graph.write_pdf(filename)


def save_file(graph, filename):
    if filename.split(".")[-1] in ("PNG", "png"):
        _save_png(graph, filename)

    elif filename.split(".")[-1] in ("PDF", "pdf"):
        _save_pdf(graph, filename)



def visualize_tree(clf, feature_name, method, depth=-1, format="png"):
    if depth == -1:
        depth = clf.max_depth
    else:
        assert 1 <= depth <= clf.max_depth, f"Please set depth from 1 to {clf.max_depth}"

    if not method == "Random Forest":
        dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_name)
        graph = pydotplus.graph_from_dot_data(dot_data)
        save_file(graph, "graph.pdf")

    else:
        for i, _clf in enumerate(clf.estimators_):
            dot_data = tree.export_graphviz(_clf, out_file=None, feature_names=feature_name)
            graph = pydotplus.graph_from_dot_data(dot_data)
            save_file(graph, f"graph{i}.pdf")