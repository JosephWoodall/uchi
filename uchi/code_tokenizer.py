"""
code_tokenizer.py
=================
Phase 1: AST-structural tokenizer for Python code.

Converts Python source to structural tokens for the trie, giving it
syntactic knowledge by construction rather than raw character patterns.
"""

import ast
from typing import List


class ASTCodeTokenizer:
    """Convert Python source to structural tokens suitable for trie training."""

    def tokenize_source(self, source: str) -> List[str]:
        """Tokenize Python source into structural tokens."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source.split()
        return _walk_module(tree)

    def tokenize_query(self, query_words: list) -> List[str]:
        """Map natural-language code query words to structural hints."""
        mapping = {
            "def": "DEF", "function": "DEF", "func": "DEF", "method": "DEF",
            "class": "CLASS", "object": "CLASS",
            "import": "IMPORT", "from": "FROM",
            "return": "RETURN", "returns": "RETURN",
            "if": "IF", "else": "ELSE", "elif": "IF", "when": "IF", "condition": "IF",
            "for": "FOR", "loop": "FOR", "iterate": "FOR", "each": "FOR",
            "while": "WHILE",
            "list": "LIST", "array": "LIST",
            "dict": "DICT", "map": "DICT", "dictionary": "DICT",
            "try": "TRY", "except": "EXCEPT", "error": "TRY",
            "lambda": "LAMBDA",
            "assign": "ASSIGN",
        }
        out = []
        for word in query_words:
            w = word.lower().strip(".,!?")
            out.append(mapping.get(w, word))
        return out


def _walk_module(tree: ast.Module) -> List[str]:
    tokens = []
    for node in tree.body:
        tokens.extend(_walk_node(node))
    return tokens


def _walk_node(node) -> List[str]:  # noqa: C901
    tokens = []

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        tokens.append("DEF")
        tokens.append(f"FNAME:{node.name}")
        for arg in node.args.args:
            tokens.append(f"ARG:{arg.arg}")
        if node.returns:
            tokens.append("RETURN_TYPE")
        for child in node.body:
            tokens.extend(_walk_node(child))

    elif isinstance(node, ast.ClassDef):
        tokens.append(f"CLASS:{node.name}")
        for child in node.body:
            tokens.extend(_walk_node(child))

    elif isinstance(node, ast.Return):
        tokens.append("RETURN")
        if node.value:
            tokens.extend(_walk_expr(node.value))

    elif isinstance(node, ast.If):
        tokens.append("IF")
        tokens.extend(_walk_expr(node.test))
        for child in node.body:
            tokens.extend(_walk_node(child))
        if node.orelse:
            tokens.append("ELSE")
            for child in node.orelse:
                tokens.extend(_walk_node(child))

    elif isinstance(node, ast.For):
        tokens.append("FOR")
        tokens.extend(_walk_expr(node.iter))
        for child in node.body:
            tokens.extend(_walk_node(child))

    elif isinstance(node, ast.While):
        tokens.append("WHILE")
        tokens.extend(_walk_expr(node.test))
        for child in node.body:
            tokens.extend(_walk_node(child))

    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
        tokens.append("ASSIGN")
        if isinstance(node, ast.Assign):
            tokens.extend(_walk_expr(node.value))

    elif isinstance(node, ast.AugAssign):
        tokens.append("AUGASSIGN")

    elif isinstance(node, ast.Expr):
        tokens.extend(_walk_expr(node.value))

    elif isinstance(node, ast.Import):
        for alias in node.names:
            tokens.append(f"IMPORT:{alias.name.split('.')[0]}")

    elif isinstance(node, ast.ImportFrom):
        if node.module:
            tokens.append(f"FROM:{node.module.split('.')[0]}")

    elif isinstance(node, ast.Try):
        tokens.append("TRY")
        for child in node.body:
            tokens.extend(_walk_node(child))
        if node.handlers:
            tokens.append("EXCEPT")
        for child in node.finalbody if hasattr(node, 'finalbody') else []:
            tokens.extend(_walk_node(child))

    elif isinstance(node, ast.With):
        tokens.append("WITH")
        for child in node.body:
            tokens.extend(_walk_node(child))

    elif isinstance(node, ast.Raise):
        tokens.append("RAISE")

    elif isinstance(node, ast.Delete):
        tokens.append("DEL")

    elif isinstance(node, ast.Pass):
        tokens.append("PASS")

    elif isinstance(node, ast.Break):
        tokens.append("BREAK")

    elif isinstance(node, ast.Continue):
        tokens.append("CONTINUE")

    elif isinstance(node, ast.Global):
        tokens.append("GLOBAL")

    elif isinstance(node, ast.Nonlocal):
        tokens.append("NONLOCAL")

    elif isinstance(node, ast.Assert):
        tokens.append("ASSERT")

    return tokens


def _walk_expr(expr) -> List[str]:
    tokens = []

    if isinstance(expr, ast.Call):
        if isinstance(expr.func, ast.Name):
            tokens.append(f"CALL:{expr.func.id}")
        elif isinstance(expr.func, ast.Attribute):
            tokens.append(f"CALL:{expr.func.attr}")
        else:
            tokens.append("CALL")
        for arg in expr.args:
            tokens.extend(_walk_expr(arg))

    elif isinstance(expr, ast.Name):
        tokens.append(f"NAME:{expr.id}")

    elif isinstance(expr, ast.Constant):
        if isinstance(expr.value, str):
            tokens.append("CONST_STR")
        elif isinstance(expr.value, bool):
            tokens.append("CONST_BOOL")
        elif isinstance(expr.value, (int, float)):
            tokens.append("CONST_NUM")
        elif expr.value is None:
            tokens.append("CONST_NONE")
        else:
            tokens.append("CONST")

    elif isinstance(expr, ast.BinOp):
        tokens.append("BINOP")
        tokens.extend(_walk_expr(expr.left))
        tokens.extend(_walk_expr(expr.right))

    elif isinstance(expr, ast.UnaryOp):
        tokens.append("UNARYOP")

    elif isinstance(expr, ast.Compare):
        tokens.append("COMPARE")

    elif isinstance(expr, ast.BoolOp):
        tokens.append("BOOLOP")

    elif isinstance(expr, ast.List):
        tokens.append("LIST")

    elif isinstance(expr, ast.Dict):
        tokens.append("DICT")

    elif isinstance(expr, ast.Set):
        tokens.append("SET")

    elif isinstance(expr, (ast.Tuple,)):
        tokens.append("TUPLE")

    elif isinstance(expr, ast.ListComp):
        tokens.append("LISTCOMP")

    elif isinstance(expr, ast.DictComp):
        tokens.append("DICTCOMP")

    elif isinstance(expr, ast.SetComp):
        tokens.append("SETCOMP")

    elif isinstance(expr, ast.GeneratorExp):
        tokens.append("GENEXP")

    elif isinstance(expr, ast.Lambda):
        tokens.append("LAMBDA")

    elif isinstance(expr, ast.IfExp):
        tokens.append("TERNARY")

    elif isinstance(expr, ast.Attribute):
        tokens.append(f"ATTR:{expr.attr}")

    elif isinstance(expr, ast.Subscript):
        tokens.append("SUBSCRIPT")

    elif isinstance(expr, ast.Starred):
        tokens.append("STARRED")

    elif isinstance(expr, ast.JoinedStr):
        tokens.append("FSTRING")

    return tokens
