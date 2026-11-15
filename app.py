"""Entry point alias. The CLI lives in scripts/recommend.py.

Usage: python app.py --user-id 1 --n 10
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.recommend import main

if __name__ == "__main__":
    main()
