"""
Self-contained VinePPO-style math grading.
Combines answer extraction, normalization, and sympy-based equivalence checking.
No external dependencies beyond standard library + sympy.

Based on:
- https://github.com/openai/prm800k/blob/main/prm800k/grading/grader.py
- https://github.com/McGill-NLP/VinePPO (math_grader, math_normalize, math_answer_extraction)
"""

import re
import sympy
import concurrent.futures
from typing import Optional
from sympy.parsing import sympy_parser


# ============================================================
# Answer Extraction (from model response)
# ============================================================

def _fix_fracs_extract(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        new_str += "{" + a + "}{" + b + "}" + substr[2:]
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        new_str += "{" + a + "}" + b + substr[2:]
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b_extract(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except:
        return string


def _fix_sqrt_extract(string):
    _string = re.sub(r"\\sqrt(-?[0-9.a-zA-Z]+)", r"\\sqrt{\1}", string)
    _string = re.sub(r"\\sqrt\s+(\w+)$", r"\\sqrt{\1}", _string)
    return _string


def _fix_tan_extract(string):
    _string = re.sub(r"\\tan(-?[0-9.a-zA-Z]+)", r"\\tan{\1}", string)
    _string = re.sub(r"\\tan\s+(\w+)$", r"\\tan{\1}", _string)
    return _string


def strip_string(string):
    string = str(string).strip()
    string = string.replace("\n", "")
    string = string.rstrip(".")
    string = string.replace("\\!", "")

    if string.startswith("\\text{") and string.endswith("}"):
        string = string.split("{", 1)[1][:-1]

    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("cfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    _string = re.sub(r"\\text\{.*?\}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string

    string = string.replace("^{\\circ}", "").strip()
    string = string.replace("^\\circ", "").strip()
    string = re.sub(r"\{(c|m)?m\}(\^(2|3))?", "", string).strip()
    string = re.sub(r"p\.m\.$", "", string).strip()
    string = re.sub(r"(\d)\s*t$", r"\1", string).strip()

    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("x\\in", "")
    string = string.replace("\\%", "%")
    string = string.replace("\%", "%")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    string = string.replace("\\cdot", "")
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")
    string = string.replace("\\mathbf", "")
    string = string.replace("\\mathrm", "")
    string = re.sub(r"\\mbox\{.*?\}", "", string)
    string = string.replace("'", "")
    string = string.replace('"', "")

    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    string = _fix_sqrt_extract(string)
    string = _fix_tan_extract(string)
    string = string.replace(" ", "")
    string = _fix_fracs_extract(string)
    string = _fix_a_slash_b_extract(string)
    string = re.sub(r"(\\|,|\.)+$", "", string)

    return string


def extract_boxed_answers(text):
    answers = []
    for piece in text.split("boxed{")[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == "{":
                n += 1
            elif piece[i] == "}":
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == "%":
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    return answers


def extract_answer(pred_str, exhaust=False):
    pred = []
    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = [tmp.split("$. I hope", 1)[0].strip()]
    elif "boxed" in pred_str:
        pred = extract_boxed_answers(pred_str)
    elif "he answer is" in pred_str:
        pred = [pred_str.split("he answer is")[-1].strip()]
    else:
        pattern = r"-?\d*\.?\d+"
        ans = re.findall(pattern, pred_str.replace(",", ""))
        if len(ans) >= 1:
            ans = ans[-1]
        else:
            ans = ""
        if ans:
            pred.append(ans)

    _pred = []
    for ans in pred:
        ans = ans.strip().split("\n")[0]
        ans = ans.lstrip(":")
        ans = ans.rstrip(".")
        ans = ans.rstrip("/")
        ans = strip_string(ans)
        _pred.append(ans)

    if exhaust:
        return _pred
    else:
        return _pred[-1] if _pred else ""


# ============================================================
# Answer Normalization (Hendrycks MATH style)
# ============================================================

def _fix_fracs_norm(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        new_str += "{" + a + "}{" + b + "}" + substr[2:]
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        new_str += "{" + a + "}" + b + substr[2:]
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b_norm(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except:
        return string


def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string


def _fix_sqrt_norm(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt_norm(string)
    string = string.replace(" ", "")
    string = _fix_fracs_norm(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b_norm(string)
    return string


def normalize_answer_mathd(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer


# ============================================================
# Grading (OpenAI PRM800k style)
# ============================================================

BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [
    r"\^[0-9]+\^",
    r"\^-?[0-9][0-9]+",
    r"\*\*[0-9]+\*\*",
    r"\*\*-?[0-9][0-9]+",
    r"[0-9][0-9][0-9]+!",
]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex_simple(expr: str) -> str:
    """Parse LaTeX to plain text without pylatexenc dependency."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")

    # Simple LaTeX to text conversion (replaces pylatexenc)
    expr = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"((\1)/(\2))", expr)
    expr = re.sub(r"\\sqrt\{([^}]*)\}", r"sqrt(\1)", expr)
    expr = re.sub(r"\\sqrt([0-9])", r"sqrt(\1)", expr)
    expr = re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]*)\}", r"\1", expr)
    expr = expr.replace("\\pi", "pi")
    expr = expr.replace("\\infty", "inf")
    expr = expr.replace("\\cdot", "*")
    expr = expr.replace("\\times", "*")
    expr = expr.replace("\\div", "/")
    expr = re.sub(r"\^\{([^}]*)\}", r"**(\1)", expr)
    expr = re.sub(r"\{([^}]*)\}", r"(\1)", expr)
    expr = expr.replace("\\", "")

    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _strip_properly_formatted_commas(expr: str):
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub(r"\1\3\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    p1 = re.compile(r"([0-9]) +([0-9])")
    step = p1.sub(r"\1+\2", step)
    return step


def _normalize(expr: str) -> str:
    if expr is None:
        return None

    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree", "cm", "centimeter", "meter", "mile", "second", "minute",
        "hour", "day", "week", "month", "year", "foot", "feet", "inch", "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(r",\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex_simple(expr)
        except:
            pass

    expr = re.sub(r"- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def _count_unknown_letters(expr: str):
    for word in ["sqrt", "frac", "pi", "inf", "cos", "sin", "tan", "cot",
                  "arccos", "arcsin", "arctan", "arccot", "log", "ln", "exp"]:
        expr = expr.replace(word, "")
    return len(set(x for x in expr if x.isalpha()))


def _should_allow_eval(expr: str):
    if _count_unknown_letters(expr) > 2:
        return False
    for bad in BAD_SUBSTRINGS:
        if bad in expr:
            return False
    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False
    return True


def _are_equal_under_sympy(gt_norm: str, given_norm: str, timeout: int = 10):
    def _check():
        try:
            expr = f"({gt_norm})-({given_norm})"
            if _should_allow_eval(expr):
                sympy_diff = _sympy_parse(expr)
                simplified = sympy.simplify(sympy_diff)
                if simplified == 0:
                    return True
        except:
            pass
        return False

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_check)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return False


def _split_tuple(expr: str):
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (len(expr) > 2
        and expr[0] in TUPLE_CHARS and expr[-1] in TUPLE_CHARS
        and all(ch not in expr[1:-1] for ch in TUPLE_CHARS)):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def grade_answer(*, given_answer: str = None, ground_truth: str = None) -> bool:
    """
    Grade if given_answer matches ground_truth.
    (a) normalization match, or (b) sympy equivalence.
    """
    if given_answer is None:
        return False

    gt_mathd = normalize_answer_mathd(ground_truth)
    given_mathd = normalize_answer_mathd(given_answer)
    if gt_mathd == given_mathd:
        return True

    gt_norm = _normalize(ground_truth)
    given_norm = _normalize(given_answer)

    if gt_norm is None:
        return False
    if gt_norm == given_norm:
        return True
    if len(given_norm) == 0:
        return False

    gt_elems = _split_tuple(gt_norm)
    given_elems = _split_tuple(given_norm)

    if len(gt_elems) > 1 and (
        gt_norm[0] != given_norm[0] or gt_norm[-1] != given_norm[-1]
    ):
        return False
    if len(gt_elems) != len(given_elems):
        return False

    for gt_elem, given_elem in zip(gt_elems, given_elems):
        if _is_frac(gt_elem) and _is_frac(given_elem):
            is_correct = (gt_elem == given_elem)
        elif _str_is_int(gt_elem) != _str_is_int(given_elem):
            is_correct = False
        else:
            is_correct = _are_equal_under_sympy(gt_elem, given_elem)
        if not is_correct:
            return False

    return True


# ============================================================
# Public API
# ============================================================

def vineppo_judge(response: str, ground_truth: str) -> bool:
    """
    Judge correctness using VinePPO's grading pipeline.
    Extracts answer from response, then grades against ground truth.
    """
    try:
        predicted_answer = extract_answer(response, exhaust=False)
        if not predicted_answer:
            return False
        return grade_answer(given_answer=predicted_answer, ground_truth=ground_truth)
    except Exception:
        return False
