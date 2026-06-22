#!/usr/bin/env python3
import subprocess
import sys

def main():
    print("========================================")
    print(" Running Uchi Comprehensive Test Suite  ")
    print("========================================\n")
    
    # Run pytest on the tests directory
    result = subprocess.run(["python", "-m", "pytest", "tests/", "-v"])
    
    if result.returncode == 0:
        print("\nAll tests passed successfully! No regressions detected.")
    else:
        print("\nSome tests failed. Please review the output above.")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
