#!/usr/bin/env python3
"""Cross-run stability checker (regime-split, user-approved 2026-09-25).

Thin wrapper over tools/check_stability.py (the canonical gate, kept as the
single source of the regime-split thresholds so the two cannot drift).
Usage: python3 tools/spread.py results/A.json results/B.json [results/C.json ...]
"""
import os
import sys

sys.exit(os.execv(sys.executable, [sys.executable,
                                   os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_stability.py"),
                                   "--suite=all", *sys.argv[1:]]))
