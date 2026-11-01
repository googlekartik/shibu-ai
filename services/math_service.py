"""Safe mathematics with SymPy. Never uses eval() on user input.

Supported:
  25 + 50, 25% of 800, sqrt(144), 2^10, 10 / 4
  solve x + 5 = 12, factor x^2 - 5x + 6
  derivative of x^3, integral of x^2
  mean/median of 1, 2, 3, 4
"""
import re
import statistics

import sympy as sp

x = sp.Symbol("x")

ALLOWED_CHARS = re.compile(r"^[0-9x\+\-\*/\^\%\(\)\.\s,=a-z]+$")


def _clean(expr: str) -> str:
    expr = expr.strip().lower()
    expr = re.sub(r"^(calculate|calc|compute|what is|solve|find)\s+", "", expr)
    expr = expr.rstrip("?").strip()
    expr = expr.replace("^", "**")
    return expr


UNIT_ALIASES = {
    "km": "km", "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km",
    "mile": "mile", "miles": "mile", "mi": "mile",
    "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "cm": "cm", "centimeter": "cm", "centimeters": "cm", "centimetre": "cm", "centimetres": "cm",
    "inch": "inch", "inches": "inch", "in": "inch",
    "foot": "foot", "feet": "foot", "ft": "foot",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "gram": "gram", "grams": "gram", "g": "gram",
    "pound": "pound", "pounds": "pound", "lb": "pound", "lbs": "pound",
    "ounce": "ounce", "ounces": "ounce", "oz": "ounce",
    "litre": "litre", "litres": "litre", "liter": "litre", "liters": "litre", "l": "litre",
    "millilitre": "ml", "millilitres": "ml", "milliliter": "ml", "milliliters": "ml", "ml": "ml",
    "gallon": "gallon", "gallons": "gallon",
    "celsius": "celsius", "c": "celsius", "fahrenheit": "fahrenheit", "f": "fahrenheit",
    "kelvin": "kelvin", "k": "kelvin",
}
LENGTH_TO_M = {"km": 1000.0, "mile": 1609.34, "m": 1.0, "cm": 0.01, "inch": 0.0254, "foot": 0.3048}
WEIGHT_TO_KG = {"kg": 1.0, "gram": 0.001, "pound": 0.453592, "ounce": 0.0283495}
VOLUME_TO_L = {"litre": 1.0, "ml": 0.001, "gallon": 3.78541}


def _fmt_num(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return ("%.6f" % value).rstrip("0").rstrip(".")


def _convert(value: float, from_unit: str, to_unit: str):
    """Return converted value, or None if the units are incompatible."""
    if from_unit in ("celsius", "fahrenheit", "kelvin") or to_unit in ("celsius", "fahrenheit", "kelvin"):
        to_c = {"celsius": value, "fahrenheit": (value - 32) * 5 / 9, "kelvin": value - 273.15}[from_unit]
        return {"celsius": to_c, "fahrenheit": to_c * 9 / 5 + 32, "kelvin": to_c + 273.15}[to_unit]
    for table in (LENGTH_TO_M, WEIGHT_TO_KG, VOLUME_TO_L):
        if from_unit in table and to_unit in table:
            return value * table[from_unit] / table[to_unit]
    return None


def _annuity_parse(text: str):
    """Parse a regular-savings sentence: a monthly deposit, an
    annual interest rate, and a number of years.

    e.g. 'I save 500 rupees a month at 12 percent annual interest
    compounded monthly. How much do I have after 2 years?'
    Returns (deposit, annual_rate_percent, years) or None.
    """
    low = text.lower()
    if not re.search(r"\b(save|deposit|invest|put aside)\b", low):
        return None
    if not re.search(r"\b(per month|a month|monthly|every month)\b", low):
        return None
    if not re.search(r"\b(interest|percent|%)\b", low):
        return None
    dep = re.search(r"(?:save|deposit|invest|put aside)\s+(?:rs\.?|rupees?|₹|\$)?\s*(\d+(?:\.\d+)?)", low)
    if not dep:
        dep = re.search(r"(\d+(?:\.\d+)?)\s*(?:rs\.?|rupees?|₹|\$)?\s*(?:per month|a month|monthly|every month)", low)
    rate = re.search(r"(\d+(?:\.\d+)?)\s*(?:percent|%|per\s*cent)", low)
    yrs = re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", low)
    if not dep or not rate or not yrs:
        return None
    return float(dep.group(1)), float(rate.group(1)), float(yrs.group(1))


def looks_like_math(text: str) -> bool:
    low = text.lower().strip()
    if _annuity_parse(text) is not None:
        return True
    if re.search(r"\bconvert\b.*\bto\b", low):
        return True
    if re.match(r"^(calculate|calc|compute)\b", low):
        return True
    if re.search(r"\b(derivative|integral|solve|factor|sqrt|mean|median)\b", low):
        return True
    if re.search(r"\d+\s*%\s*of\s*\d+", low):
        return True
    if re.search(r"\d+\s*[\+\-\*/\^]\s*\d+", low) and not re.search(r"[a-z]{3,}", low.replace("sqrt", "")):
        return True
    if re.match(r"^[\d\s\+\-\*/\^\%\(\)\.]+$", low) and any(c.isdigit() for c in low):
        return True
    if re.search(r"\bsolve\b.*=", low):
        return True
    return False


class MathService:
    def calculate(self, raw: str):
        try:
            # Regular savings (annuity): monthly deposit + annual
            # rate + years -> future value, before SymPy parsing
            # (a sentence is not an arithmetic expression).
            annuity = _annuity_parse(raw)
            if annuity is not None:
                deposit, rate_pct, years = annuity
                r = rate_pct / 100.0 / 12.0
                n = int(round(years * 12))
                fv = deposit * (((1 + r) ** n - 1) / r) if r > 0 else deposit * n
                return {"success": True, "expression": raw.strip(),
                        "calculation": (f"Monthly deposit {_fmt_num(deposit)}, annual rate {rate_pct}% "
                                        f"compounded monthly: monthly rate r = {rate_pct}/100/12 = {r:.6f}, "
                                        f"months n = {years:g} x 12 = {n}; "
                                        f"FV = {_fmt_num(deposit)} x ((1 + {r:.6f})^{n} - 1) / {r:.6f}"),
                        "answer": _fmt_num(fv)}

            expr_text = _clean(raw)

            # Percentage: "25% of 800"
            m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", expr_text)
            if m:
                pct, base = float(m.group(1)), float(m.group(2))
                ans = pct / 100.0 * base
                ans = int(ans) if ans.is_integer() else ans
                return {"success": True, "expression": raw.strip(),
                        "calculation": f"{pct}% of {base} = {pct}/100 x {base}",
                        "answer": str(ans)}

            # Unit conversion: "convert 5 km to miles", "convert 100 celsius to fahrenheit"
            m = re.search(r"convert\s+(\d+(?:\.\d+)?)\s*([a-z]+)\s+to\s+([a-z]+)", expr_text)
            if m:
                value = float(m.group(1))
                from_unit = UNIT_ALIASES.get(m.group(2))
                to_unit = UNIT_ALIASES.get(m.group(3))
                if not from_unit or not to_unit:
                    raise ValueError("Unsupported unit. Try km/miles, kg/pounds, cm/inches, litres/gallons, celsius/fahrenheit.")
                result = _convert(value, from_unit, to_unit)
                if result is None:
                    raise ValueError("Those units cannot be converted into each other.")
                ans = _fmt_num(result)
                return {"success": True, "expression": raw.strip(),
                        "calculation": f"{_fmt_num(value)} {from_unit} = {ans} {to_unit}",
                        "answer": ans}

            # Derivative
            m = re.search(r"derivative of (.+)", expr_text)
            if m:
                expr = sp.sympify(m.group(1).strip(), locals={"x": x})
                deriv = sp.diff(expr, x)
                return {"success": True, "expression": raw.strip(),
                        "calculation": f"d/dx({expr})", "answer": str(deriv)}

            # Integral
            m = re.search(r"integral of (.+)", expr_text)
            if m:
                expr = sp.sympify(m.group(1).strip(), locals={"x": x})
                integ = sp.integrate(expr, x)
                return {"success": True, "expression": raw.strip(),
                        "calculation": f"∫({expr}) dx", "answer": str(integ)}

            # Solve equation
            if "=" in expr_text and "solve" in raw.lower() or ("=" in expr_text and "x" in expr_text):
                left, right = expr_text.split("=", 1)
                eq = sp.Eq(sp.sympify(left.strip(), locals={"x": x}),
                           sp.sympify(right.strip(), locals={"x": x}))
                sol = sp.solve(eq, x)
                return {"success": True, "expression": raw.strip(),
                        "calculation": f"Solve {eq}", "answer": str(sol)}

            # Factor
            m = re.search(r"factor (.+)", expr_text)
            if m:
                expr = sp.sympify(m.group(1).strip(), locals={"x": x})
                return {"success": True, "expression": raw.strip(),
                        "calculation": f"factor({expr})", "answer": str(sp.factor(expr))}

            # Statistics
            m = re.search(r"(mean|median|mode) of ([\d,\s\.]+)", expr_text)
            if m:
                nums = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", m.group(2))]
                if not nums:
                    raise ValueError("No numbers found.")
                fn = {"mean": statistics.mean, "median": statistics.median,
                      "mode": statistics.mode}[m.group(1)]
                ans = fn(nums)
                return {"success": True, "expression": raw.strip(),
                        "calculation": f"{m.group(1)}({nums})", "answer": str(ans)}

            # General arithmetic / sqrt / powers, via SymPy only
            if not ALLOWED_CHARS.match(expr_text):
                raise ValueError("Expression contains unsupported characters.")
            expr = sp.sympify(expr_text, locals={"x": x, "sqrt": sp.sqrt})
            result = sp.simplify(expr)
            # Present clean numbers: 5 not 5.00000000000000, 0.5 not 0.5000.
            if result.is_number:
                try:
                    f = float(sp.N(result))
                    if abs(f - round(f)) < 1e-9:
                        ans = str(int(round(f)))
                    else:
                        ans = ("%.10f" % f).rstrip("0").rstrip(".")
                except Exception:
                    ans = str(result)
            else:
                ans = str(result)
            return {"success": True, "expression": raw.strip(),
                    "calculation": f"{expr_text} = {ans}", "answer": ans}
        except Exception as exc:
            return {"success": False, "expression": raw.strip(),
                    "calculation": "", "answer": "", "error": str(exc)}
