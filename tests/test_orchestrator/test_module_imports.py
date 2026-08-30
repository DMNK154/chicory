"""Guards against imports that `from __future__ import annotations` hides.

Annotations are never evaluated, so a module can type a signature with
`np.ndarray`, use `np.*` at runtime, and still import cleanly — the
NameError only surfaces when that specific handler is finally called.
`librarian_recall` shipped broken for exactly this reason.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_MODULES = [
    "chicory/orchestrator/orchestrator.py",
    "chicory/layer4/librarian.py",
    "chicory/layer4/desk.py",
    "chicory/layer4/ingress.py",
]


def _runtime_names(tree: ast.AST) -> set[str]:
    """Root names used in real code, ignoring annotations and strings."""
    used: set[str] = set()

    class V(ast.NodeVisitor):
        def visit_AnnAssign(self, node):
            # Skip the annotation; visit the value only
            if node.value:
                self.visit(node.value)

        def visit_arg(self, node):
            return  # annotations on parameters never evaluate

        def visit_FunctionDef(self, node):
            for child in node.body:
                self.visit(child)
            for d in node.decorator_list:
                self.visit(d)

        def visit_Attribute(self, node):
            n = node
            while isinstance(n, ast.Attribute):
                n = n.value
            if isinstance(n, ast.Name):
                used.add(n.id)
            self.generic_visit(node)

    V().visit(tree)
    return used


@pytest.mark.parametrize("relpath", _MODULES)
def test_runtime_numpy_use_has_a_module_import(relpath):
    path = pathlib.Path(__file__).resolve().parents[2] / relpath
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if "np" not in _runtime_names(tree):
        pytest.skip(f"{relpath} does not use np at runtime")

    module_level = {
        alias.asname or alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "np" in module_level, (
        f"{relpath} calls np.* at runtime but never imports numpy at module "
        "level — annotations hide this until the call is made"
    )
