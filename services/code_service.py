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
    ("python", "javascript"): ("Python is a general-purpose, dynamically typed programming language that runs on servers, in data tools and in scripts, and is widely used for web development backends. JavaScript is a dynamically typed language that runs in every web browser and, with Node.js, also on servers, and is the main language of front-end web development. Python web frameworks include Django and Flask, which handle routing, databases and templates on the server. JavaScript web frameworks include React, Angular and Vue on the front end, with Node.js and Express on the back end. Python is often chosen for data-heavy backends and APIs, while JavaScript is chosen when the same language must run in the browser and on the server. Trade-off: a rich data and backend ecosystem (Python) vs the only language that runs natively in the browser (JavaScript)."),
}

# Hand-written quizzes for comparisons, so "quiz me on that
# comparison" tests the comparison itself — never a web search.
# Each question: {question, options, answer} with answer in options.
COMPARISON_QUIZZES = {
    ("python", "javascript"): [
        {"question": "Which language runs in every web browser?",
         "options": ["JavaScript", "Python", "Both Python and JavaScript"],
         "answer": "JavaScript"},
        {"question": "Which Python web framework is batteries-included?",
         "options": ["Django", "React", "Express"],
         "answer": "Django"},
        {"question": "Which language runs on servers with Node.js?",
         "options": ["JavaScript", "Python", "SQL"],
         "answer": "JavaScript"},
    ],
}


JS_TEMPLATES = {
    "factorial": '''function factorial(n) {
  let result = 1;
  for (let i = 2; i <= n; i++) {
    result *= i;
  }
  return result;
}

console.log(factorial(5)); // 120''',
    "fibonacci": '''function fibonacci(n) {
  const out = [];
  let a = 0, b = 1;
  for (let i = 0; i < n; i++) {
    out.push(a);
    [a, b] = [b, a + b];
  }
  return out;
}

console.log(fibonacci(10));''',
    "reverse string": '''function reverseString(s) {
  return s.split("").reverse().join("");
}

console.log(reverseString("Shibu")); // "ubihS"''',
    "sort": '''const numbers = [5, 2, 9, 1, 7];
numbers.sort((a, b) => a - b);
console.log(numbers); // [1, 2, 5, 7, 9]''',
    "prime": '''function isPrime(n) {
  if (n < 2) return false;
  for (let i = 2; i <= Math.sqrt(n); i++) {
    if (n % i === 0) return false;
  }
  return true;
}

console.log(isPrime(7)); // true''',
    "even odd": '''const number = 7;
if (number % 2 === 0) {
  console.log("Even");
} else {
  console.log("Odd");
}''',
    "palindrome": '''function isPalindrome(s) {
  const clean = s.toLowerCase().replace(/ /g, "");
  return clean === clean.split("").reverse().join("");
}

console.log(isPalindrome("madam")); // true''',
}

# Words that carry no task meaning in a code request. Removing them
# lets "reverse a string" match the "reverse string" template, and
# "write python code to find if a number is prime" match "prime".
_CODE_STOPWORDS = frozenset({
    "a", "an", "the", "write", "code", "python", "program", "please", "me",
    "generate", "create", "show", "for", "to", "in", "using", "with", "that",
    "can", "you", "give", "example", "of", "and", "explain", "it", "now",
    "same", "javascript", "js", "java", "script", "function", "check",
    "find", "is", "if", "number", "numbers", "or", "how", "do", "i",
})
# Single trigger words that name a task on their own.
_TASK_ALIASES = {
    "sorting": "sort", "sorted": "sort", "reverse": "reverse string",
    "palindromes": "palindrome", "factorials": "factorial",
}


def detect_language(request: str) -> str:
    """'javascript' when the request names JavaScript/JS, else 'python'."""
    low = (request or "").lower()
    if re.search(r"\bjavascript\b|\bjs\b|java\s*script", low):
        return "javascript"
    return "python"


def match_task_key(request: str):
    """Return the template key for a code request, or None.

    Matches the exact phrase first ("reverse string"), then the
    stopword-normalised word set ("reverse a string" -> reverse +
    string), so articles and filler words never block a match.
    """
    low = (request or "").lower()
    for key in CODE_TEMPLATES:
        if key in low:
            return key
    words = set(re.findall(r"[a-z]+", low))
    for alias, key in _TASK_ALIASES.items():
        if alias in words:
            return key
    # Singular forms too: "palindromes" -> "palindrome".
    expanded = set(words)
    for w in words:
        if w.endswith("s") and len(w) > 3:
            expanded.add(w[:-1])
    content = expanded - _CODE_STOPWORDS
    for key in CODE_TEMPLATES:
        if all(w in content for w in key.split()):
            return key
    return None


def generate_code_for(task_key: str, language: str = "python"):
    """Return (code, explanation) for a template key and language."""
    if language == "javascript":
        code = JS_TEMPLATES.get(task_key)
        label = "JavaScript"
    else:
        code = CODE_TEMPLATES.get(task_key)
        label = "Python"
    if code is None:
        return None, None
    return code, (f"Here is {label} code for {task_key}. "
                  "It is broken into small steps so you can follow it line by line.")


# Per-line explanations, one bullet per key line, so a code answer
# never claims to be "broken into small steps" without showing the
# steps. Bullets name the exact operations in the template above.
_CODE_EXPLANATIONS = {
    "factorial": {
        "python": [
            "`result = 1` starts the running product at 1.",
            "`for i in range(2, n + 1)` walks through 2, 3, and so on up to n.",
            "`result *= i` multiplies the running product by each number.",
            "`return result` gives back n! — `print(factorial(5))` shows 120.",
        ],
        "javascript": [
            "`let result = 1` starts the running product at 1.",
            "`for (let i = 2; i <= n; i++)` walks through 2, 3, and so on up to n.",
            "`result *= i` multiplies the running product by each number.",
            "`return result` gives back n! — `console.log(factorial(5))` shows 120.",
        ],
    },
    "fibonacci": {
        "python": [
            "`a, b = 0, 1` starts the sequence with 0 and 1.",
            "Each loop appends `a` to the list, then moves on with `a, b = b, a + b`.",
            "`return out` gives back the first n Fibonacci numbers.",
        ],
        "javascript": [
            "`let a = 0, b = 1` starts the sequence with 0 and 1.",
            "Each loop pushes `a`, then moves on with `[a, b] = [b, a + b]`.",
            "`return out` gives back the first n Fibonacci numbers.",
        ],
    },
    "reverse string": {
        "python": [
            "`s[::-1]` slices the string backwards, from the end to the start.",
            "`return` gives back the reversed string.",
            "`print(reverse_string(\"Shibu\"))` shows \"ubihS\".",
        ],
        "javascript": [
            "`s.split(\"\")` turns the string into a list of characters.",
            "`.reverse()` reverses that list, and `.join(\"\")` joins it back into a string.",
            "`console.log(reverseString(\"Shibu\"))` shows \"ubihS\".",
        ],
    },
    "sort": {
        "python": [
            "`numbers.sort()` sorts the list in place, smallest first.",
            "`print(numbers)` shows [1, 2, 5, 7, 9].",
        ],
        "javascript": [
            "`numbers.sort((a, b) => a - b)` sorts numerically, smallest first (the comparator is needed: without it, sort compares as text).",
            "`console.log(numbers)` shows [1, 2, 5, 7, 9].",
        ],
    },
    "prime": {
        "python": [
            "Numbers below 2 are not prime, so they return `False` straight away.",
            "The loop tries divisors from 2 up to the square root of n (`int(n ** 0.5) + 1`).",
            "If any divisor divides n evenly (`n % i == 0`), n is not prime.",
            "Otherwise the function returns `True`.",
        ],
        "javascript": [
            "Numbers below 2 are not prime, so they return `false` straight away.",
            "The loop tries divisors from 2 up to the square root of n (`Math.sqrt(n)`).",
            "If any divisor divides n evenly (`n % i === 0`), n is not prime.",
            "Otherwise the function returns `true`.",
        ],
    },
    "even odd": {
        "python": [
            "`number % 2` is the remainder after dividing by 2.",
            "A remainder of 0 prints \"Even\"; any other remainder prints \"Odd\".",
        ],
        "javascript": [
            "`number % 2` is the remainder after dividing by 2.",
            "A remainder of 0 logs \"Even\"; any other remainder logs \"Odd\".",
        ],
    },
    "palindrome": {
        "python": [
            "Lowercasing and removing spaces makes the check ignore case and spacing.",
            "`s == s[::-1]` is True only when the cleaned string reads the same backwards.",
            "`print(is_palindrome(\"madam\"))` shows True.",
        ],
        "javascript": [
            "Lowercasing and removing spaces makes the check ignore case and spacing.",
            "`clean === clean.split(\"\").reverse().join(\"\")` is true only when the cleaned string reads the same backwards.",
            "`console.log(isPalindrome(\"madam\"))` shows true.",
        ],
    },
}


def explain_code(task_key: str, language: str = "python") -> str:
    """Bullet list explaining each key line of a template, or ""."""
    bullets = (_CODE_EXPLANATIONS.get(task_key) or {}).get(language) or []
    return "\n".join(f"- {b}" for b in bullets)


def detect_code_request(request: str, last_task=None):
    """Return (task_key, language) for a code request, or None.

    Handles 'Write Python/JavaScript code to/for ...', 'code for ...',
    and follow-ups like 'now write the same in JavaScript' (which
    reuse the previous task in the new language).
    """
    low = (request or "").lower()
    language = detect_language(request)
    key = match_task_key(request)
    if key:
        return key, language
    if last_task and re.search(r"\bwrite\b", low) and re.search(r"\b(same|it|that)\b", low):
        return last_task, language
    return None


def generate_code(request: str):
    """Return (code, explanation) or (None, None) if no template matches."""
    key = match_task_key(request)
    if key is None:
        return None, None
    return generate_code_for(key, detect_language(request))


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


def _normalise_compare_name(name: str) -> str:
    """'JavaScript for web development in a table' -> 'javascript';
    'JS' -> 'javascript'. Trailing purpose phrases are not part of
    the compared entity."""
    text = (name or "").strip().lower().strip(" ?.")
    text = re.split(r"\s+(?:for|in|as|with|to)\b", text, maxsplit=1)[0].strip()
    text = re.sub(r"\s+(?:table|comparison|compared)$", "", text).strip()
    if text in ("js", "java script"):
        return "javascript"
    if text == "py":
        return "python"
    return text


def compare(a: str, b: str) -> str:
    key = (_normalise_compare_name(a), _normalise_compare_name(b))
    rev = (key[1], key[0])
    if key in COMPARISONS:
        return COMPARISONS[key]
    if rev in COMPARISONS:
        return COMPARISONS[rev]
    return (f"I don't have a ready-made comparison for {a} vs {b} yet, so I won't guess one. "
            f"To compare them fairly, look at: 1) What is each one best at? 2) How easy is it to learn/use? "
            f"3) How fast / scalable is it? 4) What are its main downsides? "
            f"Try a ready comparison instead: 'Compare Python and Java' or 'Compare Flask and Django'.")
