import contextlib
import io
import math
import ast
import re
import collections
import functools
import itertools
import multiprocessing as mp
import queue as queue_module
import statistics
from fractions import Fraction
from typing import Any, Dict, Iterable, Sequence

from config import TOOL_TIMEOUT_SECONDS, TOOL_MAX_CODE_CHARS, TOOL_MAX_OUTPUT_CHARS


ALLOWED_IMPORTS = {
    "math": math,
    "collections": collections,
    "functools": functools,
    "itertools": itertools,
    "statistics": statistics,
    "fractions": __import__("fractions"),
}

try:
    import sympy
except ImportError:  # pragma: no cover - optional dependency
    sympy = None
else:
    ALLOWED_IMPORTS["sympy"] = sympy

try:
    import numpy
except ImportError:  # pragma: no cover - optional dependency
    numpy = None
else:
    ALLOWED_IMPORTS["numpy"] = numpy


def _safe_import(name: str, globals=None, locals=None, fromlist=(), level=0):
    if level != 0:
        raise ImportError("relative_import_not_allowed")
    root_name = name.split(".", 1)[0]
    module = ALLOWED_IMPORTS.get(root_name)
    if module is None:
        raise ImportError(f"import_not_allowed: {name}")
    return module


def vp(n: int, p: int) -> int | float:
    if p <= 1:
        raise ValueError("p_must_be_prime_like")
    value = abs(int(n))
    if value == 0:
        return math.inf
    exponent = 0
    while value % p == 0:
        value //= p
        exponent += 1
    return exponent


def legendre_vp_factorial(n: int, p: int) -> int:
    if n < 0:
        raise ValueError("n_must_be_nonnegative")
    if p <= 1:
        raise ValueError("p_must_be_prime_like")
    total = 0
    q = p
    while q <= n:
        total += n // q
        q *= p
    return total


def small_mod_check(values: Iterable[int], modulus: int) -> tuple[int, ...]:
    if modulus <= 0:
        raise ValueError("modulus_must_be_positive")
    return tuple(int(value) % modulus for value in values)


def factorint_small(n: int) -> dict[int, int]:
    value = abs(int(n))
    factors: dict[int, int] = {}
    if value <= 1:
        return factors
    count = 0
    while value % 2 == 0:
        value //= 2
        count += 1
    if count:
        factors[2] = count
    p = 3
    while p * p <= value:
        count = 0
        while value % p == 0:
            value //= p
            count += 1
        if count:
            factors[p] = count
        p += 2
    if value > 1:
        factors[value] = factors.get(value, 0) + 1
    return factors


def divisors_small(n: int) -> tuple[int, ...]:
    factors = factorint_small(n)
    divisors = [1]
    for prime, exponent in factors.items():
        current = list(divisors)
        prime_power = 1
        for _ in range(exponent):
            prime_power *= prime
            divisors.extend(value * prime_power for value in current)
    return tuple(sorted(set(divisors)))


def digitsum_base(n: int, base: int) -> int:
    if base <= 1:
        raise ValueError("base_must_be_greater_than_one")
    value = abs(int(n))
    if value == 0:
        return 0
    total = 0
    while value:
        value, digit = divmod(value, base)
        total += digit
    return total


def base_reduction_path(n: int, base: int) -> tuple[int, ...]:
    if base <= 1:
        raise ValueError("base_must_be_greater_than_one")
    value = abs(int(n))
    path = [value]
    while value >= base:
        value = digitsum_base(value, base)
        path.append(value)
    return tuple(path)


def bounded_search(candidates: Iterable[Any], predicate, limit: int = 100000) -> Any | None:
    if limit <= 0:
        raise ValueError("limit_must_be_positive")
    for index, candidate in enumerate(candidates, start=1):
        if index > limit:
            raise ValueError("search_limit_exceeded")
        if predicate(candidate):
            return candidate
    return None


def small_convolution(a: Sequence[int | float], b: Sequence[int | float]) -> list[int | float]:
    if not a or not b:
        return []
    result = [0] * (len(a) + len(b) - 1)
    for i, left in enumerate(a):
        for j, right in enumerate(b):
            result[i + j] += left * right
    return result


def poly_eval(coeffs: Sequence[int | float], x: int | float) -> int | float:
    value = 0
    for coeff in coeffs:
        value = value * x + coeff
    return value


def packing_feasibility(rectangles: Iterable[Sequence[int]], width: int, height: int) -> dict[str, Any]:
    container_area = int(width) * int(height)
    total_area = 0
    individually_feasible = True
    normalized_rectangles: list[tuple[int, int]] = []
    for rectangle in rectangles:
        if len(rectangle) != 2:
            raise ValueError("rectangles_must_have_two_dimensions")
        w, h = int(rectangle[0]), int(rectangle[1])
        normalized_rectangles.append((w, h))
        total_area += w * h
        if not ((w <= width and h <= height) or (h <= width and w <= height)):
            individually_feasible = False
    return {
        "container_area": container_area,
        "total_area": total_area,
        "area_ok": total_area <= container_area,
        "individually_feasible": individually_feasible,
        "slack_area": container_area - total_area,
        "rectangles": normalized_rectangles,
    }


def dist2(point_a: Sequence[int | float], point_b: Sequence[int | float]) -> int | float:
    ax, ay = point_a
    bx, by = point_b
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


def orientation(
    point_a: Sequence[int | float],
    point_b: Sequence[int | float],
    point_c: Sequence[int | float],
) -> int | float:
    ax, ay = point_a
    bx, by = point_b
    cx, cy = point_c
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def line_intersection(
    point_a: Sequence[int | float],
    point_b: Sequence[int | float],
    point_c: Sequence[int | float],
    point_d: Sequence[int | float],
) -> tuple[Fraction, Fraction] | None:
    ax, ay = Fraction(point_a[0]), Fraction(point_a[1])
    bx, by = Fraction(point_b[0]), Fraction(point_b[1])
    cx, cy = Fraction(point_c[0]), Fraction(point_c[1])
    dx, dy = Fraction(point_d[0]), Fraction(point_d[1])

    denominator = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)
    if denominator == 0:
        return None

    determinant_ab = ax * by - ay * bx
    determinant_cd = cx * dy - cy * dx
    x = (determinant_ab * (cx - dx) - (ax - bx) * determinant_cd) / denominator
    y = (determinant_ab * (cy - dy) - (ay - by) * determinant_cd) / denominator
    return (x, y)


SAFE_GLOBALS = {
    "__builtins__": {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "Fraction": Fraction,
        "__import__": _safe_import,
    },
    "Fraction": Fraction,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "math": math,
    "max": max,
    "min": min,
    "vp": vp,
    "legendre_vp_factorial": legendre_vp_factorial,
    "small_mod_check": small_mod_check,
    "factorint_small": factorint_small,
    "divisors_small": divisors_small,
    "digitsum_base": digitsum_base,
    "base_reduction_path": base_reduction_path,
    "bounded_search": bounded_search,
    "small_convolution": small_convolution,
    "poly_eval": poly_eval,
    "packing_feasibility": packing_feasibility,
    "dist2": dist2,
    "orientation": orientation,
    "line_intersection": line_intersection,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

FORBIDDEN_NODES = (
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.Global,
    ast.Nonlocal,
    ast.Delete,
)

FORBIDDEN_CALL_NAMES = {
    "eval",
    "exec",
    "open",
    "compile",
    "input",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
}


def _validate_code(code: str) -> str | None:
    if len(code) > TOOL_MAX_CODE_CHARS:
        return "tool_code_too_large"

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax_error: {exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_name = alias.name.split(".", 1)[0]
                if root_name not in ALLOWED_IMPORTS:
                    return f"forbidden_import: {alias.name}"
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            root_name = module_name.split(".", 1)[0]
            if root_name not in ALLOWED_IMPORTS:
                return f"forbidden_import: {module_name or '<relative>'}"
        if isinstance(node, FORBIDDEN_NODES):
            return f"forbidden_construct: {type(node).__name__}"
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return "dunder_name_not_allowed"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALL_NAMES:
                return f"forbidden_call: {node.func.id}"

    return None


def _exec_worker(code: str, max_output_chars: int, queue: mp.Queue) -> None:
    stdout_buffer = io.StringIO()
    namespace: Dict[str, Any] = {
        key: value
        for key, value in SAFE_GLOBALS.items()
    }
    initial_namespace_keys = set(namespace)

    def serialize_value(value: Any) -> str:
        try:
            return repr(value)[:400]
        except Exception:
            return f"<unreprable {type(value).__name__}>"

    def serialize_locals() -> Dict[str, str]:
        # Sort keys so JSON/log snapshots remain deterministic across runs.
        return {
            k: serialize_value(namespace[k])
            for k in sorted(namespace)
            if not k.startswith("__") and k not in initial_namespace_keys
        }

    try:
        with contextlib.redirect_stdout(stdout_buffer):
            # Use a shared namespace for globals and locals so functions,
            # comprehensions, and generator expressions can resolve names
            # defined earlier in the same tool block.
            exec(code, namespace, namespace)

        stdout = stdout_buffer.getvalue().strip()[:max_output_chars]
        queue.put({
            "ok": True,
            "stdout": stdout,
            "locals": serialize_locals(),
            "error": None,
        })
    except Exception as exc:
        stdout = stdout_buffer.getvalue().strip()[:max_output_chars]
        queue.put({
            "ok": False,
            "stdout": stdout,
            "locals": serialize_locals(),
            "error": str(exc),
        })


def execute_python(code: str, timeout_seconds: int = TOOL_TIMEOUT_SECONDS) -> Dict[str, Any]:
    validation_error = _validate_code(code)
    if validation_error:
        return {
            "ok": False,
            "stdout": "",
            "locals": {},
            "error": validation_error,
        }

    result_queue: mp.Queue = mp.Queue()
    process = mp.Process(target=_exec_worker, args=(code, TOOL_MAX_OUTPUT_CHARS, result_queue))
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "ok": False,
            "stdout": "",
            "locals": {},
            "error": "timeout",
        }

    try:
        result = result_queue.get_nowait()
    except queue_module.Empty:
        return {
            "ok": False,
            "stdout": "",
            "locals": {},
            "error": "tool_execution_no_result",
        }

    return result


_PROBLEM_KEYWORD_RE = re.compile(r"[A-Za-z]{4,}")
_COMMENT_RE = re.compile(r"^\s*#\s?(.*)$", re.MULTILINE)
_FENCED_CODE_RE = re.compile(r"^\s*```(?:python)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)
_STOPWORDS = {
    "find",
    "show",
    "that",
    "with",
    "from",
    "such",
    "have",
    "this",
    "there",
    "where",
    "which",
    "what",
    "prove",
    "compute",
    "integer",
    "answer",
}


def normalize_tool_code(code: str) -> str:
    normalized = _FENCED_CODE_RE.sub("", code or "").strip()
    return normalized


def extract_math_keywords(problem_text: str, *, limit: int = 8) -> tuple[str, ...]:
    keywords: list[str] = []
    for match in _PROBLEM_KEYWORD_RE.finditer(problem_text or ""):
        token = match.group(0).lower()
        if token in _STOPWORDS:
            continue
        if token not in keywords:
            keywords.append(token)
        if len(keywords) >= limit:
            break
    return tuple(keywords)


def code_semantic_warning(problem_text: str, code: str) -> str | None:
    keywords = extract_math_keywords(problem_text)
    if not keywords:
        return None
    comment_text = " ".join(match.group(1).lower() for match in _COMMENT_RE.finditer(code or ""))
    code_text = f"{comment_text}\n{code or ''}".lower()
    hits = [keyword for keyword in keywords if keyword in code_text]
    if hits:
        return None
    preview = ", ".join(keywords[:4])
    return f"semantic_warning: no_problem_keywords_matched ({preview})"


def _is_retryable_tir_error(error_text: str | None) -> bool:
    if not isinstance(error_text, str) or not error_text:
        return False
    return (
        error_text.startswith("syntax_error:")
        or "is not defined" in error_text
        or error_text.startswith("import_not_allowed:")
    )


_EXPLICIT_IMPORT_RE = re.compile(r"^\s*(?:import\s+\w+|from\s+\w+(?:\.\w+)*\s+import\b)", re.MULTILINE)


def _retry_prelude(code: str) -> str:
    if _EXPLICIT_IMPORT_RE.search(code or ""):
        return code

    prelude_lines = [
        "import math",
        "import itertools",
        "import collections",
        "from math import gcd, factorial, floor, ceil, sqrt, log",
        "from itertools import combinations, permutations, product, chain",
        "from collections import defaultdict, Counter",
        "from functools import lru_cache",
        "from fractions import Fraction",
    ]
    if sympy is not None:
        prelude_lines.append("import sympy")
    if numpy is not None:
        prelude_lines.append("import numpy")
    return "\n".join(prelude_lines + ["", code]).strip()


def execute_python_with_tir(
    code: str,
    *,
    timeout_seconds: int = TOOL_TIMEOUT_SECONDS,
    problem_text: str = "",
    tir_emphasis: str = "",
) -> Dict[str, Any]:
    normalized_code = normalize_tool_code(code)
    semantic_warning = code_semantic_warning(problem_text, normalized_code)
    result = execute_python(normalized_code, timeout_seconds=timeout_seconds)
    retry_count = 0

    if not result.get("ok") and _is_retryable_tir_error(result.get("error")):
        retry_count = 1
        retried_code = _retry_prelude(normalized_code)
        retried_result = execute_python(retried_code, timeout_seconds=timeout_seconds)
        if retried_result.get("ok") or not result.get("stdout"):
            result = retried_result

    return {
        **result,
        "tir_retry_count": retry_count,
        "tir_semantic_warning": semantic_warning,
        "tir_emphasis": tir_emphasis or None,
    }
