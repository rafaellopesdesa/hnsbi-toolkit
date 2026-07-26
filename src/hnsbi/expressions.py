"""Safe arithmetic expressions for parameter-dependent sample multipliers."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

_FUNCTIONS = {
    "abs": np.abs,
    "clip": np.clip,
    "exp": np.exp,
    "log": np.log,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "sqrt": np.sqrt,
}

_BINARY = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.Pow: lambda left, right: left**right,
}

_UNARY = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}


class ExpressionError(ValueError):
    """Raised for unsafe or malformed multiplier expressions."""


@dataclass(frozen=True)
class Expression:
    """A parsed expression evaluated without Python ``eval``.

    The language contains numeric literals, named parameters, arithmetic, and
    the explicitly listed NumPy functions. Attribute access, subscripting,
    comprehensions, comparisons, and arbitrary calls are rejected.
    """

    source: str
    _tree: ast.Expression
    names: frozenset[str]

    @classmethod
    def parse(cls, source: str | float | int) -> Expression:
        source = str(source)
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError as exc:
            raise ExpressionError(f"Invalid expression {source!r}.") from exc
        names: set[str] = set()
        cls._validate(tree.body, names)
        names.difference_update(_FUNCTIONS)
        return cls(source=source, _tree=tree, names=frozenset(names))

    @classmethod
    def _validate(cls, node: ast.AST, names: set[str]) -> None:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ExpressionError("Only numeric constants are allowed.")
            return
        if isinstance(node, ast.Name):
            if node.id.startswith("_"):
                raise ExpressionError("Private names are not allowed.")
            names.add(node.id)
            return
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            cls._validate(node.left, names)
            cls._validate(node.right, names)
            if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
                if abs(float(node.right.value)) > 32:
                    raise ExpressionError("Constant powers must have |p| <= 32.")
            return
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            cls._validate(node.operand, names)
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in _FUNCTIONS:
                raise ExpressionError(f"Function {node.func.id!r} is not allowed.")
            if node.keywords:
                raise ExpressionError("Keyword arguments are not allowed.")
            for argument in node.args:
                cls._validate(argument, names)
            return
        raise ExpressionError(f"Unsupported expression element: {type(node).__name__}.")

    def evaluate(self, parameters: Mapping[str, Any]) -> Any:
        missing = self.names.difference(parameters)
        if missing:
            raise ExpressionError(
                f"Expression {self.source!r} is missing parameters {sorted(missing)}."
            )
        return self._evaluate_node(self._tree.body, parameters)

    @classmethod
    def _evaluate_node(cls, node: ast.AST, parameters: Mapping[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in _FUNCTIONS:
                return _FUNCTIONS[node.id]
            return parameters[node.id]
        if isinstance(node, ast.BinOp):
            operation = _BINARY[type(node.op)]
            return operation(
                cls._evaluate_node(node.left, parameters),
                cls._evaluate_node(node.right, parameters),
            )
        if isinstance(node, ast.UnaryOp):
            return _UNARY[type(node.op)](cls._evaluate_node(node.operand, parameters))
        if isinstance(node, ast.Call):
            function = _FUNCTIONS[node.func.id]
            return function(
                *[cls._evaluate_node(argument, parameters) for argument in node.args]
            )
        raise AssertionError(f"Validated node {type(node).__name__} was lost.")

    def simple_normfactors(self) -> tuple[str, ...] | None:
        """Return a product of bare parameter names when representable.

        This is the exact subset that maps to the upstream
        ``nsbi-common-utils`` ``normfactor`` modifiers.
        """

        def flatten(node: ast.AST) -> list[str] | None:
            if isinstance(node, ast.Name) and node.id not in _FUNCTIONS:
                return [node.id]
            if isinstance(node, ast.Constant) and float(node.value) == 1.0:
                return []
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                left = flatten(node.left)
                right = flatten(node.right)
                if left is None or right is None:
                    return None
                return left + right
            return None

        result = flatten(self._tree.body)
        if result is None or len(result) != len(set(result)):
            # The pinned upstream model de-duplicates modifier names per
            # sample, so ``mu * mu`` would be evaluated incorrectly as
            # ``mu``. Keep powers and repeated factors on the formula route.
            return None
        return tuple(result)
