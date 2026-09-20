"""Canonical synthetic-pair entry point.

The implementation lives in generate_pairs_2.py.  This compatibility
wrapper keeps the documented command stable while avoiding two active
generators with different data contracts.
"""

from generate_pairs_2 import main


if __name__ == "__main__":
    main()
