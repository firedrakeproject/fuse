"""Update code written for the old FUSE API.

* Removes the third positional argument (G2) of every ``DOFGenerator(...)`` call.
* Renames the attribute ``.g1`` to ``.transversal``.
* Removes the third space (the pullback) from the spaces of every
  ``ElementTriple(cell, (V, W_I, F), ...)`` call written as a tuple or list.
* Renames the keyword ``pullback=`` of ``.eval(...)`` calls to ``trace=``.
* Removes the trace argument ``TrH1``, ``TrHCurl`` or ``TrHDiv`` from
  ``immerse(...)`` calls: the trace is read from each kernel's action. For
  ``TrHCurl`` and ``TrHDiv`` this is reported, because kernels that declared no
  action used to take it from the trace and now need ``action=HCurl`` or
  ``action=HDiv``.

Positions come from the syntax tree, so calls split over several lines,
nested calls and calls inside lists are handled, and the rest of the source
is left exactly as it was.

Usage::

    python scripts/drop_g2.py [--check] PATH [PATH ...]

PATH is a file or a directory (searched for ``*.py``). With ``--check`` the
files are not changed; the edits that would be made are listed. Anything that
cannot be rewritten safely is reported for a manual look: ``DOFGenerator``
calls with keywords or an unexpected number of arguments, ``ElementTriple``
calls whose spaces are not written out, the names ``Fid``, ``Fcurl``,
``Fdiv``, the pullback classes and the removed traces, and ``.spaces[2]``.
"""
import argparse
import ast
import pathlib
import sys


_PULLBACK_NAMES = {"Fid", "Fcurl", "Fdiv", "Pullback", "IdentityPullback", "CovariantPiola", "ContravariantPiola"}
_TRACE_NAMES = {"TrH1", "TrHCurl", "TrHDiv"}
_REMOVED_NAMES = _PULLBACK_NAMES | _TRACE_NAMES


def _name(func):
    """The name a call is made through: ``f`` for ``f(...)`` and ``obj.f(...)``."""
    return getattr(func, "id", getattr(func, "attr", None))


def _offsets(src):
    """Turn (line, column) positions from the syntax tree into string offsets."""
    lines = src.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(lineno, col):
        # ast columns count UTF-8 bytes
        return starts[lineno - 1] + len(lines[lineno - 1].encode("utf-8")[:col].decode("utf-8"))
    return offset


def rewrite(src):
    """The updated source and a list of (line, description) of the edits."""
    tree = ast.parse(src)
    offset = _offsets(src)
    edits, report, uses = [], [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _name(node.func) == "DOFGenerator":
            if len(node.args) == 3 and not node.keywords:
                second, third = node.args[1], node.args[2]
                edits.append((offset(second.end_lineno, second.end_col_offset),
                              offset(third.end_lineno, third.end_col_offset), ""))
                report.append((node.lineno, f"removed G2 argument {ast.get_source_segment(src, third)}"))
            elif len(node.args) != 2 or node.keywords:
                report.append((node.lineno, "DOFGenerator call left for a manual look "
                               f"({len(node.args)} positional arguments, keywords {[k.arg for k in node.keywords]})"))
        elif isinstance(node, ast.Call) and _name(node.func) == "ElementTriple":
            spaces = node.args[1] if len(node.args) > 1 else None
            if isinstance(spaces, (ast.Tuple, ast.List)) and len(spaces.elts) == 3:
                second, third = spaces.elts[1], spaces.elts[2]
                edits.append((offset(second.end_lineno, second.end_col_offset),
                              offset(third.end_lineno, third.end_col_offset), ""))
                report.append((node.lineno, f"removed pullback {ast.get_source_segment(src, third)}"))
            elif not (isinstance(spaces, (ast.Tuple, ast.List)) and len(spaces.elts) == 2):
                report.append((node.lineno, "ElementTriple call left for a manual look (spaces not written out as a pair or triple)"))
        elif isinstance(node, ast.Call) and _name(node.func) == "immerse":
            traces = [(i, a) for i, a in enumerate(node.args) if i == 2 and isinstance(a, ast.Name) and a.id in _TRACE_NAMES]
            traces += [(None, k) for k in node.keywords if k.arg == "target_space"
                       and isinstance(k.value, ast.Name) and k.value.id in _TRACE_NAMES]
            for i, arg in traces:
                name = arg.id if i is not None else arg.value.id
                if i is not None:
                    previous = node.args[1]
                    start = offset(previous.end_lineno, previous.end_col_offset)
                    end = offset(arg.end_lineno, arg.end_col_offset)
                else:
                    previous = node.args[-1] if node.args else None
                    start = offset(previous.end_lineno, previous.end_col_offset)
                    end = offset(arg.value.end_lineno, arg.value.end_col_offset)
                edits.append((start, end, ""))
                note = "" if name == "TrH1" else f": check the triple's kernels declare action={name[2:]}"
                report.append((node.lineno, f"removed trace {name}{note}"))
        elif isinstance(node, ast.Call) and _name(node.func) == "eval":
            for kw in node.keywords:
                if kw.arg == "pullback":
                    start = offset(kw.lineno, kw.col_offset)
                    edits.append((start, start + len("pullback"), "trace"))
                    report.append((kw.lineno, "renamed eval(pullback=) to eval(trace=)"))
        elif isinstance(node, ast.Attribute) and node.attr == "g1":
            end = offset(node.end_lineno, node.end_col_offset)
            edits.append((end - len("g1"), end, "transversal"))
            report.append((node.lineno, "renamed .g1 to .transversal"))
        elif isinstance(node, ast.Name) and node.id in _REMOVED_NAMES:
            uses.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module == "fuse.spaces.pullbacks":
            report.append((node.lineno, "imports fuse.spaces.pullbacks, which has been removed: remove by hand"))
        elif isinstance(node, ast.alias) and node.name in _REMOVED_NAMES:
            report.append((node.lineno, f"imports {node.name}, which has been removed: remove by hand"))
        elif (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "spaces"
              and isinstance(node.slice, ast.Constant) and node.slice.value == 2):
            report.append((node.lineno, ".spaces[2] (the pullback) has been removed: use .pullback() instead"))
    removed = [(start, end) for start, end, text in edits if text == ""]
    for node in uses:
        at = offset(node.lineno, node.col_offset)
        if not any(start <= at < end for start, end in removed):
            report.append((node.lineno, f"uses {node.id}, which has been removed: remove by hand"))
    for start, end, text in sorted(edits, reverse=True):
        src = src[:start] + text + src[end:]
    ast.parse(src)
    return src, sorted(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=pathlib.Path)
    parser.add_argument("--check", action="store_true", help="list the edits without changing any file")
    args = parser.parse_args(argv)
    files = []
    for path in args.paths:
        files += sorted(path.rglob("*.py")) if path.is_dir() else [path]
    for path in files:
        src = path.read_text()
        new, report = rewrite(src)
        if not report:
            continue
        if new != src and not args.check:
            path.write_text(new)
        print(path)
        for lineno, description in report:
            print(f"  line {lineno}: {description}")


if __name__ == "__main__":
    sys.exit(main())
