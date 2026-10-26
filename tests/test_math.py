import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.math_service import MathService

svc = MathService()


def test_arithmetic():
    assert svc.calculate("2 + 3")["answer"] == "5"


def test_percentage():
    assert svc.calculate("25% of 800")["answer"] == "200"


def test_sqrt():
    assert svc.calculate("sqrt(144)")["answer"] == "12"


def test_derivative():
    assert "3" in svc.calculate("derivative of x^3")["answer"]
