"""
Test Suite Runner for Pharmacy Medicine Demand & Expiry-Risk Forecasting.
ALGIONS AI LABS - Assessment Project

Runs all unit, regression, leakage, model, and API test suites.
"""
import sys
import unittest
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_all_tests():
    print("=" * 80)
    print("RUNNING PHARMACY DECISION-SUPPORT SOLUTION TEST SUITE")
    print("ALGIONS AI LABS - Verification & Acceptance Testing")
    print("=" * 80)
    
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: Ran {result.testsRun} tests in {elapsed:.2f} seconds.")
    if result.wasSuccessful():
        print("STATUS: ALL TESTS PASSED SUCCESSFULLY! (100% PASS RATE)")
        print("=" * 80)
        return 0
    else:
        print(f"STATUS: FAILED with {len(result.failures)} failures and {len(result.errors)} errors.")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

