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
    # Standard markdown code block - prioritizing this (handles python, py, or no tag)
    m = re.search(r"```(?:python|py)?\s*\n?(.*?)```", llm_output, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    # Generic code block (sometimes the model forgets the first backticks or similar)
    m = re.search(r"```\s*\n?(.*?)```", llm_output, re.DOTALL)
    if m:
        return m.group(1).strip()
    
    # Fallback: if no markdown blocks, try to identify code lines directly.
    # We look for lines starting with key Python keywords.
    lines = llm_output.splitlines()
    code_lines = []
    in_code_zone = False
    
    for line in lines:
        clean_line = line.strip()
        # Markers that strongly suggest code start
        if any(clean_line.startswith(p) for p in ["import ", "from ", "def ", "class ", "x =", "y =", "n =", "result =", "PYTHON_OPTIONS"]):
            in_code_zone = True
        
        if in_code_zone:
            # Simple heuristic: stop if we hit something that looks like conversational text
            # and we already have some code lines.
            if not clean_line and code_lines: # skip empty lines but keep going
                code_lines.append(line)
                continue
                
            if code_lines and any(clean_line.startswith(p) for p in ["The ", "To ", "We ", "This ", "So "]):
                # If we've already seen a print statement, we can likely stop.
                if any("print(" in l for l in code_lines):
                    break
            
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()

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
