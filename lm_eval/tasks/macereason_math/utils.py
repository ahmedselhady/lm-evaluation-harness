from __future__ import annotations

import logging
import re
import signal
from importlib.metadata import version
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import datasets


eval_logger = logging.getLogger(__name__)


try:
    import antlr4  # noqa: F401
    import sympy
    from math_verify import parse, verify
    from sympy.parsing.latex import parse_latex

    assert version("antlr4-python3-runtime").startswith("4.11")
except (ModuleNotFoundError, AssertionError) as e:
    raise type(e)(
        "`sympy`, `math_verify` and `antlr4-python3-runtime==4.11` are required for generating translation task prompt templates. "
        "Please install the required packages via pip install lm-eval[math] or pip install -e .[math]"
    ) from e


# taken from
# https://github.com/wellecks/lm-evaluation-harness/blob/master/lm_eval/tasks/minerva_math.py
def doc_to_text(doc: dict) -> str:
    return (
        "Solve the problem and show concise reasoning. End with exactly one line "
        "in this format: `Final Answer: <answer>`. Put only the answer after the "
        "colon, and do not write anything after that line.\n\n"
        "Problem:\n"
        + doc["problem"]
        + "\n\nSolution:"
    )


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        solution = doc["solution"]
        boxed_answer = last_boxed_only_string(solution)
        answer = doc.get("answer")
        if boxed_answer is not None:
            answer = remove_boxed(boxed_answer)
        elif not answer:
            # Some mAceReason solutions have no boxed answer. Prefer the
            # dataset's answer field; otherwise use text after a final-answer
            # marker when available.
            match = re.search(
                r"(?:final\s+answer|answer)\s*:\s*(.+)", solution, re.IGNORECASE
            )
            answer = match.group(1).strip() if match else ""

        out_doc = {
            "problem": doc["problem"],
            "solution": solution,
            "answer": normalize_final_answer(str(answer)),
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc

    return dataset.map(_process_doc)


def list_fewshot_samples() -> list[dict]:
    return [
        {
            "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.",
            "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: $[2,5)$",
            "few_shot": "1",
        },
        {
            "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
            "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: $24$",
            "few_shot": "1",
        },
        {
            "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
            "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts $2\\cdot12\\cdot20=480$ pounds in total. With two 15-pound weights, $n$ lifts give $30n$ pounds. Thus $30n=480$, so $n=\\boxed{16}$.\nFinal Answer: $16$",
            "few_shot": "1",
        },
        {
            "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
            "solution": "Multiplying the first equation by $-\\frac{3}{2}$ gives $6y-9x=-\\frac{3}{2}a$. Since $6y-9x=b$, we get $b=-\\frac{3}{2}a$, hence $\\frac{a}{b}=\\boxed{-\\frac{2}{3}}$.\nFinal Answer: $-\\frac{2}{3}$",
            "few_shot": "1",
        },
        {
            "problem": "Solve $2x+3=11$ for $x$.",
            "solution": "Subtract 3 to get $2x=8$, then divide by 2 to obtain $x=\\boxed{4}$.\nFinal Answer: $4$",
            "few_shot": "1",
        },
    ]


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    candidates = results[0]

    unnormalized_answer = get_unnormalized_answer(candidates)
    answer = normalize_final_answer(unnormalized_answer)

    if is_equiv(answer, doc["answer"]):
        retval = 1
    else:
        retval = 0

    # math_verify
    _mvres = verify(
        gold=parse(doc["solution"]),
        target=parse(candidates),
    )
    mathval = 1 if _mvres else 0

    res = {
        "exact_match": retval,
        "math_verify": mathval,
    }
    return res


def last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    """x1 and x2 are normalized latex string."""
    # parse_latex raises on tuples, intervals and matrices, and the handler
    # below then returns False for the pair -- so without this an answer
    # identical to the gold is scored wrong. Only ever turns a False into a
    # True: identical strings sympy can parse already compare equal below.
    if isinstance(x1, str) and x1 and x1 == x2:
        return True

    try:
        with timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                eval_logger.debug("couldn't parse one of %s or %s", x1, x2)
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                eval_logger.debug("couldn't subtract %s and %s", x1, x2)
                return False

            try:
                return sympy.simplify(diff) == 0
            except ValueError:
                eval_logger.debug(
                    "Had some trouble simplifying when comparing %s and %s", x1, x2
                )
                return False
    except TimeoutError:
        eval_logger.debug("Timed out comparing %s and %s", x1, x2)
        return False
    except ImportError as e:
        eval_logger.error(e)
        raise
    except Exception as e:  # noqa: BLE001
        eval_logger.debug("Failed comparing %s and %s with %s", x1, x2, e)
        return False


def get_unnormalized_answer(text: str) -> str:
    INVALID_ANSWER = "[invalidanswer]"
    match = re.search(
        r"Final Answer:\s*The final answer is(.*?)(?:\.\s*I hope it is correct\.|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    match = re.search(r"Final Answer:\s*(.+?)\s*(?:\n|$)", text, re.IGNORECASE)
    return match.group(1).strip() if match else INVALID_ANSWER


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    # Remove unit words while whitespace still delimits them, so that
    # "2\pi cm" loses "cm" and "\left" keeps its "ft" (word boundaries
    # cannot match inside a LaTeX command name)
    for expr in REMOVED_EXPRESSIONS:
        if expr.isalpha():
            final_answer = re.sub(rf"(?<!\\)\b{expr}\b", "", final_answer)

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        if expr.isalpha():
            # Units glued to their surroundings once whitespace is gone;
            # skip LaTeX command spans so "ft" cannot corrupt "\left".
            # Trade-off: a zero-separator <command><unit> run like "\picm"
            # survives as one token instead of being stripped by accident.
            segments = re.split(r"(\\[a-zA-Z]+)", final_answer)
            for i in range(0, len(segments), 2):
                segments[i] = segments[i].replace(expr, "")
            final_answer = "".join(segments)
        else:
            final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    # \sqrt[3]a -> \sqrt[3]{a}: canonicalize indexed roots with an unbraced
    # argument before the shorthand rule below, which must skip them entirely
    # or its [^ {] class would consume the "[" of the index and corrupt
    # \sqrt[3]{8} into \sqrt{[}3]{8}.
    final_answer = re.sub(r"(sqrt)(\[[^\]]*\])(\w)", "\\1\\2{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)(?!\[)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000: strip commas only when they are true
    # thousands-group separators. A bare tuple like "0,1" would otherwise
    # pass the old digit check and get fused into the garbage number "01"
    # (16 MATH test-set gold answers are bare tuples).
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", final_answer):
        final_answer = final_answer.replace(",", "")

    return final_answer
