from __future__ import annotations

import ast
import copy
from typing import TYPE_CHECKING, Protocol

try:
    import sympy as sp
except ImportError:  # pragma: no cover - exercised via runtime fallback
    sp = None

if TYPE_CHECKING:
    from mot_sr.code_manipulation import Function


class FunctionLike(Protocol):
    name: str
    args: str
    body: str
    return_type: str | None

    def __str__(self) -> str:
        ...


_COMMON_SYMPY_FUNCS = {
    "sin": lambda: sp.sin,
    "cos": lambda: sp.cos,
    "tan": lambda: sp.tan,
    "asin": lambda: sp.asin,
    "acos": lambda: sp.acos,
    "atan": lambda: sp.atan,
    "sinh": lambda: sp.sinh,
    "cosh": lambda: sp.cosh,
    "tanh": lambda: sp.tanh,
    "exp": lambda: sp.exp,
    "log": lambda: sp.log,
    "sqrt": lambda: sp.sqrt,
    "Abs": lambda: sp.Abs,
    "abs": lambda: sp.Abs,
    "sign": lambda: sp.sign,
}


def raw_function_ast_length(function: FunctionLike) -> int:
    return _ast_length(str(function))


def simplified_function_ast_length(function: FunctionLike) -> int:
    """Return AST length after SymPy simplification, with raw-AST fallback."""
    fallback = raw_function_ast_length(function)
    if sp is None:
        return fallback

    try:
        simplified_expr = _simplified_expression(function)
        if simplified_expr is None:
            return fallback
        return _ast_length(_canonical_wrapper(function, simplified_expr))
    except Exception:
        return fallback


def _ast_length(code: str) -> int:
    tree = ast.parse(code)
    return sum(1 for _ in ast.walk(tree))


def _simplified_expression(function: FunctionLike) -> str | None:
    parsed = ast.parse(_function_source(function))
    if not parsed.body or not isinstance(parsed.body[0], ast.FunctionDef):
        return None

    function_node = parsed.body[0]
    resolved_expr = _resolve_return_expression(function_node)
    if resolved_expr is None:
        return None

    sympy_expr = sp.sympify(
        ast.unparse(resolved_expr),
        locals=_sympy_locals(function_node),
    )
    simplified_expr = sp.simplify(sympy_expr)
    return str(simplified_expr)


def _function_source(function: FunctionLike) -> str:
    return_type = f" -> {function.return_type}" if function.return_type else ""
    return f"def {function.name}({function.args}){return_type}:\n{function.body}\n"


def _canonical_wrapper(function: FunctionLike, simplified_expr: str) -> str:
    return_type = f" -> {function.return_type}" if function.return_type else ""
    return (
        f"def {function.name}({function.args}){return_type}:\n"
        f"    return {simplified_expr}\n"
    )


def _resolve_return_expression(function_node: ast.FunctionDef) -> ast.expr | None:
    aliases: dict[str, ast.expr] = {}

    for stmt in function_node.body:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                return None
            aliases[stmt.targets[0].id] = _substitute_aliases(stmt.value, aliases)
            continue

        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                return None
            return _substitute_aliases(stmt.value, aliases)

        return None

    return None


def _substitute_aliases(expr: ast.expr, aliases: dict[str, ast.expr]) -> ast.expr:
    class AliasSubstituter(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            replacement = aliases.get(node.id)
            if replacement is None:
                return node
            return copy.deepcopy(replacement)

    return AliasSubstituter().visit(copy.deepcopy(expr))


def _sympy_locals(function_node: ast.FunctionDef) -> dict[str, object]:
    locals_map: dict[str, object] = {
        "np": sp,
        "numpy": sp,
    }

    for arg in function_node.args.args:
        if arg.arg == "params":
            locals_map[arg.arg] = sp.IndexedBase(arg.arg)
        else:
            locals_map[arg.arg] = sp.Symbol(arg.arg)

    for name, factory in _COMMON_SYMPY_FUNCS.items():
        locals_map[name] = factory()

    return locals_map
