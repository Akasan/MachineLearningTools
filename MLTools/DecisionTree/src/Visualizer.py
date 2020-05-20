from graphviz import Digraph
import pydotplus
from sklearn.externals.six import StringIO
from sklearn import tree


# sklearn標準の可視化手法を使う

def visualize_tree(clf, feature_name, depth=-1):
    if depth == -1:
        depth = clf.max_depth
    else:
        assert 1 <= depth <= clf.max_depth, f"Please set depth from 1 to {clf.max_depth}"

    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_name)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("graph.pdf")