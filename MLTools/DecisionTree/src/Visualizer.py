from graphviz import Digraph
import pydotplus
from sklearn.externals.six import StringIO
from sklearn import tree
from ._SaveGraph import write_file


# sklearn標準の可視化手法を使う

def _save_png(graph, filename):
    """ save tree as png file

    Arguments:
    ----------
        graph {pydotplus.graphviz.Dot} -- graph data
        filename {str} -- file name

    Examples:
    ---------
        >>> _save_png(graph, "hoge.png")        # the graph will be saves as hoge.png
    """
    graph.write_png(filename)


def _save_pdf(graph, filename):
    """ save tree as pdf file

    Arguments:
    ----------
        graph {pydotplus.graphviz.Dot} -- graph data
        filename {str} -- file name

    Examples:
    ---------
        >>> _save_png(graph, "hoge.pdf")        # the graph will be saves as hoge.pdf
    """
    graph.write_pdf(filename)


def save_file(graph, filename):
    """ save graph to file. File type will be decided according to extension.

    Arguments:
    ----------
        graph {pydotplus.graphviz.Dot} -- graph data
        filename {str} -- file name

    Examples:
    ---------
        >>> save_file(graph, "hoge.png")        # the graph will be saved as hoge.png
        >>> save_file(graph, "hoge.pdf")        # the graph will be saved as hoge.pdf
    """
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
        write_file(graph, "graph.gif", "gif")

    else:
        for i, _clf in enumerate(clf.estimators_):
            dot_data = tree.export_graphviz(_clf, out_file=None, feature_names=feature_name)
            graph = pydotplus.graph_from_dot_data(dot_data)
            print(dir(graph))
            save_file(graph, f"graph{i}.pdf")