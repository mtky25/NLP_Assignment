import sys
import io
import re
import math
import statistics
import random
from typing import Optional, List

try:
    import sympy as _sympy
    _SYMPY_AVAILABLE = True
except ImportError:
    _SYMPY_AVAILABLE = False

try:
    import numpy as _numpy
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


def _extract_code(llm_output: str) -> Optional[str]:
    """Extract Python code from LLM output, trying several patterns."""
    # Standard ```python ... ``` block
    m = re.search(r"```python\s+(.*?)```", llm_output, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    # Generic ``` ... ``` block
    m = re.search(r"```\s*(.*?)```", llm_output, re.DOTALL)
    if m:
        return m.group(1)
    # If the output looks like raw Python (has print() and no markdown prose), use it as-is
    if "print(" in llm_output and not llm_output.strip().startswith(("#", "The", "To", "We", "A ")):
        return llm_output
    return None


def execute_pot_code(llm_output: str, options: Optional[List[str]] = None) -> Optional[int]:
    """
    Extracts Python code from LLM output, executes it, and returns the last printed integer (0-3).
    Injects common libraries and the options list into the execution environment.
    """
    code = _extract_code(llm_output)
    if code is None:
        return None

    output_capture = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = output_capture

    try:
        exec_globals = {
            "math": math,
            "statistics": statistics,
            "random": random,
            "PYTHON_OPTIONS": options if options else [],
        }
        if _SYMPY_AVAILABLE:
            exec_globals["sympy"] = _sympy
        if _NUMPY_AVAILABLE:
            exec_globals["numpy"] = _numpy
            exec_globals["np"] = _numpy

        exec(code, exec_globals)
    except Exception as e:
        print(f"PoT exec error: {e}", file=original_stdout)
        return None
    finally:
        sys.stdout = original_stdout

    output_str = output_capture.getvalue().strip()
    if not output_str:
        return None

    integers = re.findall(r"\b(\d+)\b", output_str)
    if integers:
        try:
            result = int(integers[-1])
            if 0 <= result <= 3:
                return result
        except ValueError:
            pass

    return None
