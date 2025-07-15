#!/usr/bin/env python3
"""
CLI entry point for nnterp package.
"""
import sys
import subprocess
import importlib.resources
import argparse


def run_tests(model_names=None, class_names=None, pytest_args=None):
    """Run the nnterp tests using pytest."""
    tests_dir = importlib.resources.files("nnterp").joinpath("tests")

    cmd = [
        "pytest",
        str(tests_dir),
        "--cache-clear",
    ]
    if model_names is not None:
        cmd += ["--model-names"] + model_names
    if class_names is not None:
        cmd += ["--class-names"] + class_names
    if pytest_args is not None:
        cmd += pytest_args
    res = subprocess.run(cmd, check=False)
    return res.returncode


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m nnterp", description="nnterp command line interface"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    test_parser = subparsers.add_parser(
        "run_tests", help="Run nnterp tests using pytest"
    )
    test_parser.add_argument(
        "--model-names",
        "-m",
        help="If provided, run tests for the given model names. If not provided, run tests for all available models.",
        default=None,
        nargs="+",
    )
    test_parser.add_argument(
        "--class-names",
        "-c",
        help="If provided, run tests on the toy models of the given classes",
        default=None,
        nargs="+",
    )

    args, unknown = parser.parse_known_args()

    if args.command == "run_tests":
        return run_tests(args.model_names, args.class_names, unknown)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
