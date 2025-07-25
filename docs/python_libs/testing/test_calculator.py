import pytest
from calculator import add, div, subtract

def test_add_simple():
    assert add(1, 2) == 3, "1 + 2 should equal 3"

def test_float_close():
    assert pytest.approx(div(1, 3), rel=1e-10) == 0.333333333333

def test_raises_with_message():
    with pytest.raises(ValueError, match="non-zero"):
        div(1, 0)
        
def test_subtract():
    assert subtract(5, 2) == 3

# # ------------- Parametrization -------------
@pytest.mark.parametrize("a, b, expected", [(1, 2, 3), (2, 3, 5), (0, 0, 0)])
def test_add_parametrized(a, b, expected):
    assert add(a, b) == expected

# # For loop -
# def test_add_multiple():
#     test_cases = [
#         (1, 2, 3),
#         (2, 3, 5),
#         (0, 0, 0)
#     ]
#     for a, b, expected in test_cases:
#         assert add(a, b) == expected

# ------------- Terminal -------------
# pytest tests/test_calc.py::test_add_simple
# pytest -k "add and not float" # keyword, searches substrings (also logical operators)
# pytest -m "integration and not slow"
# pytest --lf     # last failed
# pytest --ff     # first run the last-failed tests (from cache), then run all remaining tests.
# pytest -x       # stop after first failure
# pytest --maxfail=2
