from llmath.extract import extract_answer, extract_code
from llmath.sandbox import check_code, child_env, run_code
from llmath.templates import TEMPLATES, generate
from llmath.verify import verify


def test_last_boxed_with_nested_braces():
    text = "Önce \\boxed{1} buldum, ama doğrusu \\boxed{\\frac{12}{5}} olur."
    assert extract_answer(text).answer == "\\frac{12}{5}"
    assert extract_answer(text).method == "boxed"


def test_answer_line_and_last_number_fallbacks():
    assert extract_answer("İşlemler...\n**Cevap:** 42").answer == "42"
    assert extract_answer("Sonuç olarak toplam 17 elma ve 3 armut kalır.").method == "last_number"
    assert extract_answer("").method == "none"


def test_thinking_block_is_ignored():
    assert extract_answer("<think>belki \\boxed{9}</think> Cevap: 7").answer == "7"


def test_extract_code_takes_last_block():
    text = "```python\nprint(1)\n```\nDüzeltme:\n```python\nprint(2)\n```"
    assert extract_code(text) == "print(2)"


def test_sandbox_runs_sympy():
    result = run_code("from sympy import Rational\nprint(Rational(12, 7) + Rational(1, 7))")
    assert result.ok and result.answer == "13/7"


def test_sandbox_blocks_dangerous_code():
    assert check_code("import os\nos.listdir('.')")
    assert check_code("open('x.txt', 'w')")
    assert check_code("from sympy import sympify\nsympify(\"__import__('os')\")")
    assert run_code("import subprocess").blocked


def test_sandbox_hides_api_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gizli")
    env = child_env("C:/tmp")
    assert "GEMINI_API_KEY" not in env and "PATH" not in env
    assert check_code("import os") != []           # the program cannot even read its environment


def test_sandbox_timeout():
    result = run_code("while True:\n    pass", timeout=2)
    assert result.timed_out and not result.ok


def test_templates_are_deterministic_and_self_consistent():
    first = generate(list(TEMPLATES), per_template=3, seed=7)
    again = generate(list(TEMPLATES), per_template=3, seed=7)
    assert first == again
    assert len({row["id"] for row in first}) == len(first)
    for row in first:
        assert verify(row["gold"], row["gold"]).status == "correct"
