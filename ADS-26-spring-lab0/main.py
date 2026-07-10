import pytest


def main() -> int:
    """
    Entry point that delegates to pytest.
    Allows running the test suite via `pytest`
    or `python main.py`.
    """
    return pytest.main()


if __name__ == "__main__":
    raise SystemExit(main())
