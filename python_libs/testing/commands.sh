# ============================================================
#  Testing & Coverage — terminal commands (calculator example)
#  Run these from:  python_libs/testing/
# ============================================================

# -------- 1. Running tests --------
pytest test_calculator.py                        # run all tests in file
pytest test_calculator.py::test_add_simple       # run one specific test
pytest test_calculator.py -v                     # verbose output (see each test name)
pytest test_calculator.py -k "add"               # only tests whose name contains "add"
pytest test_calculator.py -k "add and not param" # keyword filtering with logic

# -------- 2. Failure control --------
pytest test_calculator.py -x            # stop after first failure
pytest test_calculator.py --maxfail=2   # stop after 2 failures
pytest test_calculator.py --lf          # re-run only last-failed tests
pytest test_calculator.py --ff          # failed-first, then the rest
pytest test_calculator.py --tb=short    # shorter traceback on failure

# -------- 3. Coverage (needs: pip install pytest-cov) --------
pytest --cov=calculator test_calculator.py                # basic coverage report (terminal)
pytest --cov=calculator --cov-report=term-missing test_calculator.py   # show which lines are NOT covered
pytest --cov=calculator --cov-report=html test_calculator.py           # generate htmlcov/ folder — open htmlcov/index.html

# coverage with branch analysis
pytest --cov=calculator --cov-branch --cov-report=term-missing test_calculator.py

# -------- 4. Standalone coverage (without pytest plugin) --------
coverage run -m pytest test_calculator.py   # collect coverage data
coverage report -m                          # terminal report with missing lines
coverage html                               # generate htmlcov/index.html

# -------- 5. Debugging --------
pytest test_calculator.py -s                # show print() output (no capture)
pytest test_calculator.py --pdb             # drop into debugger on failure
pytest test_calculator.py --pdb-trace       # drop into debugger at start of each test

# -------- 6. Markers & parametrize --------
pytest test_calculator.py -m "not slow"     # skip tests marked @pytest.mark.slow
pytest test_calculator.py --co              # collect-only: list tests without running

# -------- 7. Useful one-liners --------
pytest --version                            # check pytest version
pip install pytest pytest-cov               # install both pytest and coverage plugin
