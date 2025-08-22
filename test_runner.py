#!/usr/bin/env python3
"""Comprehensive test script to run simplified solver against all test cases"""

import subprocess
import sys
import time
from pathlib import Path


def run_solver_test(test_number):
    """Run the simplified solver on a specific test case"""
    base_dir = Path(__file__).parent
    test_dir = base_dir.parent / "public"

    test_input = test_dir / f"test{test_number:02d}.txt"
    expected_output_file = test_dir / f"test{test_number:02d}.out"

    # Check if files exist
    if not test_input.exists():
        return f"Test {test_number:02d}: Input file not found", 0
    if not expected_output_file.exists():
        return f"Test {test_number:02d}: Expected output file not found", 0

    try:
        # Read input and expected output
        with open(test_input, "r") as f:
            input_data = f.read()
        with open(expected_output_file, "r") as f:
            expected_output = f.read().strip()

        # Run the simplified solver with timing
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, str(base_dir / "nuruomino.py")],
            input=input_data,
            text=True,
            capture_output=True,
            timeout=2,
            cwd=str(base_dir),
        )
        end_time = time.time()
        execution_time = end_time - start_time

        if result.returncode != 0:
            return (
                f"Test {test_number:02d}: FAILED - Runtime error ({execution_time:.3f}s)\n  Error: {result.stderr.strip()}",
                execution_time,
            )

        actual_output = result.stdout.strip()

        # Compare outputs
        if actual_output == expected_output:
            return (
                f"Test {test_number:02d}: PASSED ✅ ({execution_time:.3f}s)",
                execution_time,
            )
        else:
            return (
                f"Test {test_number:02d}: FAILED ❌ - Output mismatch ({execution_time:.3f}s)\n  Expected:\n{expected_output}\n  Got:\n{actual_output}",
                execution_time,
            )

    except subprocess.TimeoutExpired:
        return f"Test {test_number:02d}: FAILED ❌ - Timeout (>2s)", 2.0
    except Exception as e:
        return f"Test {test_number:02d}: FAILED ❌ - Exception: {str(e)}", 0


def main():
    """Run all available tests"""
    print("Running comprehensive tests on simplified Nuruomino solver")
    print("=" * 60)

    # Find all available test numbers
    test_dir = Path(__file__).parent.parent / "public"
    test_files = list(test_dir.glob("test*.txt"))
    test_numbers = []

    for test_file in test_files:
        try:
            # Extract number from filename like "test-01.txt"
            test_num = int(''.join(filter(str.isdigit, test_file.stem)))
            test_numbers.append(test_num)
        except (ValueError, IndexError):
            continue

    test_numbers.sort()

    if not test_numbers:
        print("No test files found!")
        return print(f"Found {len(test_numbers)} test cases: {test_numbers}")
    print("-" * 60)

    passed = 0
    failed = 0
    results = []
    total_time = 0

    for test_num in test_numbers:
        result, execution_time = run_solver_test(test_num)
        results.append(result)
        total_time += execution_time

        if "PASSED" in result:
            passed += 1
            print(result)
        else:
            failed += 1
            print(result)

        print("-" * 40)  # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {len(test_numbers)}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")
    print(f"Success rate: {passed/len(test_numbers)*100:.1f}%")
    print(f"Total execution time: {total_time:.3f}s")
    print(f"Average time per test: {total_time/len(test_numbers):.3f}s")

    if failed > 0:
        print("\nFailed tests details:")
        print("-" * 30)
        for result in results:
            if "FAILED" in result:
                print(result)
                print()


if __name__ == "__main__":
    main()