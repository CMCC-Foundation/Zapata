import pygraphviz as pgv
from collections import namedtuple
from distutils.version import LooseVersion
import torch
from torch.autograd import Variable
import warnings

Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))

SAVED_PREFIX = "_saved_"

def get_fn_name(fn, show_attrs, max_attr_chars):
    name = str(type(fn).__name__)
    if not show_attrs:
        return name
    attrs = {}
    for attr in dir(fn):
        if not attr.startswith(SAVED_PREFIX):
            continue
        val = getattr(fn, attr)
        attr = attr[len(SAVED_PREFIX):]
        if torch.is_tensor(val):
            attrs[attr] = "[saved tensor]"
        elif isinstance(val, tuple) and any(torch.is_tensor(t) for t in val):
            attrs[attr] = "[saved tensors]"
        else:
            attrs[attr] = str(val)
    if not attrs:
        return name
    max_attr_chars = max(max_attr_chars, 3)
    col1width = max(len(k) for k in attrs.keys())
    col2width = min(max(len(str(v)) for v in attrs.values()), max_attr_chars)
    sep = "-" * max(col1width + col2width + 2, len(name))
    attrstr = '%-' + str(col1width) + 's: %' + str(col2width)+ 's'
    truncate = lambda s: s[:col2width - 3] + "..." if len(s) > col2width else s
    params = '\n'.join(attrstr % (k, truncate(str(v))) for (k, v) in attrs.items())
    return name + '\n' + sep + '\n' + params

def make_dot(var, params=None, show_attrs=False, show_saved=False, max_attr_chars=50):
    if LooseVersion(torch.__version__) < LooseVersion("1.9") and (show_attrs or show_saved):
        warnings.warn(
            "make_dot: showing grad_fn attributes and saved variables "
            "requires PyTorch version >= 1.9."
        )

    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}
    else:
        param_map = {}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='10',
                     ranksep='0.1',
                     height='0.2',
                     fontname='monospace')

    dot = pgv.AGraph(directed=True)
    dot.node_attr.update(node_attr)
    dot.graph_attr.update({'size': '12,12'})
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def get_var_name(var, name=None):
        if not name:
            name = param_map[id(var)] if id(var) in param_map else ''
        return '%s\n %s' % (name, size_to_str(var.size()))

    def add_nodes(fn):
        if torch.is_tensor(fn) or fn in seen:
            return
        seen.add(fn)

        if show_saved:
            for attr in dir(fn):
                if not attr.startswith(SAVED_PREFIX):
                    continue
                val = getattr(fn, attr)
                seen.add(val)
                attr = attr[len(SAVED_PREFIX):]
                if torch.is_tensor(val):
                    dot.add_edge(str(id(fn)), str(id(val)), dir="none")
                    dot.add_node(str(id(val)),
                                 label=get_var_name(val, attr),
                                 fillcolor='orange')
                if isinstance(val, tuple):
                    for i, t in enumerate(val):
                        if torch.is_tensor(t):
                            name = attr + '[%s]' % str(i)
                            dot.add_edge(str(id(fn)), str(id(t)), dir="none")
                            dot.add_node(str(id(t)),
                                         label=get_var_name(t, name),
                                         fillcolor='orange')

        if hasattr(fn, 'variable'):
            var = fn.variable
            seen.add(var)
            dot.add_node(str(id(var)),
                         label=get_var_name(var),
                         fillcolor='lightblue')
            dot.add_edge(str(id(var)), str(id(fn)))

        dot.add_node(str(id(fn)),
                     label=get_fn_name(fn, show_attrs, max_attr_chars),
                     fillcolor='#DDDDDD')

        if hasattr(fn, 'next_functions'):
            for u in fn.next_functions:
                if u[0] is not None:
                    dot.add_edge(str(id(u[0])), str(id(fn)))
                    add_nodes(u[0])

        if hasattr(fn, 'saved_tensors'):
            for t in fn.saved_tensors:
                seen.add(t)
                dot.add_edge(str(id(t)), str(id(fn)), dir="none")
                dot.add_node(str(id(t)),
                             label=get_var_name(t),
                             fillcolor='orange')

    def add_base_tensor(var, color='darkolivegreen1'):
        if var in seen:
            return
        seen.add(var)
        dot.add_node(str(id(var)),
                     label=get_var_name(var),
                     fillcolor=color)
        if var.grad_fn:
            add_nodes(var.grad_fn)
            dot.add_edge(str(id(var.grad_fn)), str(id(var)))
        if var._is_view():
            add_base_tensor(var._base, color='darkolivegreen3')
            dot.add_edge(str(id(var._base)), str(id(var)), style="dotted")

    if isinstance(var, tuple):
        for v in var:
            add_base_tensor(v)
    else:
        add_base_tensor(var)

    resize_graph(dot)
    return dot

def make_dot_from_trace(trace):
    raise NotImplementedError(
        "This function has been moved to pytorch core and can be found here: "
        "https://pytorch.org/docs/stable/tensorboard.html"
    )

def resize_graph(dot, size_per_element=0.15, min_size=12):
    num_rows = len(dot)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr['size'] = size_str
