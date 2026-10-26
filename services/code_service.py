"""Reasoning helpers: code generation templates, debugging, comparisons.

These are rule/template based, not an LLM: the bot is honest about that.
If a request does not match a template, it says so instead of guessing.
"""
import ast
import re

CODE_TEMPLATES = {
    "factorial": '''def factorial(n):
    """Return n! (factorial of n)."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

print(factorial(5))  # 120''',
    "fibonacci": '''def fibonacci(n):
    """Return the first n Fibonacci numbers."""
    a, b = 0, 1
    out = []
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return out

print(fibonacci(10))''',
    "reverse string": '''def reverse_string(s):
    return s[::-1]

print(reverse_string("Shibu"))  # "ubihS"''',
    "sort": '''numbers = [5, 2, 9, 1, 7]
numbers.sort()
print(numbers)  # [1, 2, 5, 7, 9]''',
    "prime": '''def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

print(is_prime(17))  # True''',
    "even odd": '''number = 7
if number % 2 == 0:
    print("Even")
else:
    print("Odd")''',
    "palindrome": '''def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]

print(is_palindrome("madam"))  # True''',
}

COMPARISONS = {
    ("python", "java"): ("Python is easier to read and faster to write, with a huge data/AI ecosystem. Java is statically typed, usually faster at runtime, and dominant in large enterprise/Android systems. Trade-off: development speed and simplicity (Python) vs performance and strict typing (Java)."),
    ("flask", "django"): ("Flask is small and flexible — you choose your own database, forms and structure. Django is batteries-included — ORM, admin panel and authentication built in. Trade-off: freedom and simplicity for small apps (Flask) vs speed of building large apps with conventions (Django)."),
    ("sql", "nosql"): ("SQL databases use fixed tables and are best for structured data and complex joins with strong consistency. NoSQL databases use flexible documents/key-values and scale out easily for huge, changing data. Trade-off: consistency and structure (SQL) vs flexibility and horizontal scale (NoSQL)."),
    ("list", "tuple"): ("Lists are mutable (you can change them) and slightly slower; tuples are immutable (cannot be changed after creation), slightly faster, and can be used as dictionary keys. Trade-off: flexibility (list) vs safety and speed (tuple)."),
}


def generate_code(request: str):
    """Return (code, explanation) or (None, None) if no template matches."""
    low = request.lower()
    for key, code in CODE_TEMPLATES.items():
        if key in low:
            return code, f"Here is Python code for {key}. It is broken into small steps so you can follow it line by line."
    return None, None


def debug_code(code: str) -> str:
    """Check Python code with ast, and explain common problems in plain language."""
    code = code.strip().strip("`")
    if not code:
        return "Paste the code after 'debug this code:' and I will check it for you."
    try:
        ast.parse(code)
    except SyntaxError as exc:
        msg = str(exc)
        hint = "Check the line above the error too — Python often reports the error on the next line."
        if "expected ':'" in msg or "expected an indented block" in msg:
            hint = "This usually means a missing colon (:) after if/for/while/def, or missing indentation on the next line."
        if "invalid syntax" in msg and "(" in code and code.count("(") != code.count(")"):
            hint = "Check your brackets — the number of ( and ) does not match."
        return f"Step-by-step check:\n1. I tried to read your code as Python.\n2. It has a syntax error: {msg} (line {exc.lineno}).\n3. Fix: {hint}"
    notes = []
    if "print" not in code and "return" not in code:
        notes.append("Your code runs without errors, but it does not print or return anything — add a print() to see the result.")
    if re.search(r"=\s*=\s*=", code):
        notes.append("You have '===' — Python uses '==' for comparison, not '==='.")
    if notes:
        return "Step-by-step check:\n1. No syntax errors found.\n2. " + "\n2. ".join(notes)
    return "Step-by-step check:\n1. No syntax errors found — Python can read this code.\n2. If the result is still wrong, check the logic: print intermediate values, test with a small input (e.g. 2, 3), and check that variables are not overwritten by mistake."


def compare(a: str, b: str) -> str:
    key = (a.strip().lower(), b.strip().lower())
    rev = (key[1], key[0])
    if key in COMPARISONS:
        return COMPARISONS[key]
    if rev in COMPARISONS:
        return COMPARISONS[rev]
    return (f"I don't have a ready-made comparison for {a} vs {b} yet, so I won't guess one. "
            f"To compare them fairly, look at: 1) What is each one best at? 2) How easy is it to learn/use? "
            f"3) How fast / scalable is it? 4) What are its main downsides? "
            f"Try a ready comparison instead: 'Compare Python and Java' or 'Compare Flask and Django'.")
