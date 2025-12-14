#!/usr/bin/env python3
"""Diagnostic helper to check importing `pypsa_simplified` from this repo.

Usage:
  python scripts/check_import.py

It prints repo and src paths, attempts to import, and shows any traceback.
"""
from pathlib import Path
import sys
import traceback


def main():
    repo = Path.cwd().resolve()
    src = (repo / "src").resolve()
    print("CWD:", repo)
    print("Resolved src:", src)
    print("sys.path contains src?", str(src) in sys.path)

    if str(src) not in sys.path:
        print("Inserting src into sys.path")
        sys.path.insert(0, str(src))

    try:
        import pypsa_simplified as pkg
        print("Imported pypsa_simplified from", getattr(pkg, '__file__', '<unknown>'))
        print("Available exports:", [k for k in getattr(pkg, '__all__', dir(pkg))][:50])
    except Exception:
        print("Import failed:")
        traceback.print_exc()


if __name__ == '__main__':
    main()
