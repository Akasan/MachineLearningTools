def _write_canon(graph, filename):
    graph.write_canon(filename)

def _write_cmap(graph, filename):
    graph.write_cmap(filename)

def _rite_cmapx(graph, filename):
    graph.write_cmapx(filename)

def _write_cmapx_np(graph, filename):
    graph.write_cmapx_np(filename)

def _write_dia(graph, filename):
    graph.write_dia(filename)

def _write_dot(graph, filename):
    graph.write_dot(filename)

def _write_fig(graph, filename):
    graph.write_dot(filename)

def _write_gd(graph, filename):
    graph.write_fig(filename)

def _write_gd2(graph, filename):
    graph.write_gd2(filename)

def _write_gif(graph, filename):
    graph.write_gif(filename)

def _write_hpgl(graph, filename):
    graph.write_hpgl(filename)

def _write_imap(graph, filename):
    graph.write_imap(filename)

def _write_imap_np(graph, filename):
    graph.write_imap_np(filename)

def _write_ismap(graph, filename):
    graph.write_ismap(filename)

def _write_jpe(graph, filename):
    graph.write_jpe(filename)

def _write_jpeg(graph, filename):
    graph.write_jpeg(filename)

def _write_jpg(graph, filename):
    graph.write_jpg(filename)

def _write_mif(graph, filename):
    graph.write_mif(filename)

def _write_mp(graph, filename):
    graph.write_mp(filename)

def _write_pcl(graph, filename):
    graph.write_pcl(filename)

def _write_pdf(graph, filename):
    graph.write_pdf(filename)

def _write_pic(graph, filename):
    graph.write_pic(filename)

def _write_plain(graph, filename):
    graph.write_plain(filename)

# def _write_plain_ext(graph, filename):
#     graph.(filename)

def _write_png(graph, filename):
    graph.write_png(filename)

def _write_ps(graph, filename):
    graph.write_ps(filename)

def _write_ps2(graph, filename):
    graph.write_ps2(filename)

def _write_raw(graph, filename):
    graph.write_raw(filename)

def _write_svg(graph, filename):
    graph.canon(filename)

def _write_svgz(graph, filename):
    graph.canon(filename)

def _write_vml(graph, filename):
    graph.canon(filename)

def _write_vmlz(graph, filename):
    graph.canon(filename)

def _write_vrml(graph, filename):
    graph.canon(filename)

def _write_vtx(graph, filename):
    graph.canon(filename)

def _write_wbmp(graph, filename):
    graph.canon(filename)

def _write_xdot(graph, filename):
    graph.canon(filename)

def _write_xlib(graph, filename):
    graph.canon(filename)


func_dict = {
    "canon": _write_canon,
    "cmap": _write_cmap,
    "cmapx": _rite_cmapx,
    "cmapx_np": _write_cmapx_np,
    "dia": _write_dia,
    "dot": _write_dot,
    "fig": _write_fig,
    "gd": _write_gd,
    "gd2": _write_gd2,
    "gif": _write_gif,
    "hpgl": _write_hpgl,
    "imap": _write_imap,
    "imap_np": _write_imap_np,
    "ismap": _write_ismap,
    "jpe": _write_jpe,
    "jpeg": _write_jpeg,
    "jpg": _write_jpg,
    "mif": _write_mif,
    "mp": _write_mp,
    "pcl": _write_pcl,
    "pdf": _write_pdf,
    "pic": _write_pic,
    "plain": _write_plain,
    # "plain_ext": _write_plain_ext,
    "png": _write_png,
    "ps": _write_ps,
    "ps2": _write_ps2,
    "raw": _write_raw,
    "svg": _write_svg,
    "svgz": _write_svgz,
    "vml": _write_vml,
    "vmlz": _write_vmlz,
    "vrml": _write_vrml,
    "vtx": _write_vtx,
    "vbmp": _write_wbmp,
    "xdot": _write_xdot,
    "xlib": _write_xlib
}


def _check_extension(filename):
    """ check file type

    Arguments:
    ----------
        filename {str} -- file name

    Returns:
    --------
        {str} -- file extension

    Examples:
    ---------
        >>> _check_extension("hoge")
        None
        >>> _check_extension("hoge.png")
        'png'
    """
    split_filename = filename.split(".")
    if len(split_filename) == 1:
        return None

    return split_filename[-1].lower()

def write_file(graph, filename, mode=None):
    """ write file

    Arguments:
    ----------
        graph {pydotplus.graphviz.Dot} -- graph
        filename {str} -- file name

    Keyword Arguments:
    ------------------
        mode {str} -- type of function (default: None}

    Raises:
    -------
        Exception: this raise when write function cannot be loaded

    Examples:
    ---------
        >>> write_file(graph, "hoge.png")               # graph will be saved as hoge.png
        >>> write_file(graph, "hoge.png")               # graph will be saved as hoge.png
        >>> write_file(graph, "hoge", mode="png")       # the same result
    """
    ext = _check_extension(filename)
    if mode is None and ext is None:
        raise Exception("There're no matching extension.")

    elif mode is None and not ext is None:
        mode = ext

    filename += f".{mode}"
    func_dict[mode](graph, filename)


def get_format_list():
    """ get write function list

    Returns:
    --------
        {list} -- function list
    """
    return list(func_dict.keys())