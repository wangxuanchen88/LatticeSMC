# Copied from the earlier harness: <earlier-harness>/edge_import.py (working tree; that repo has no git commit).
# sha256 of the source file: 87a1816b06d46733d52e9a14f966656ebee3097dfffc53df79b15fadd559e26f; EDGE pin 17c3428669ed6733edd9d8c66f7dc62060b8e46d.
# Modifications: none (ROOT now resolves to the lattice_smc repo).
"""Make the vendored EDGE package importable by its own top-level module names
(EDGE code does `from dataset...`, `from model...`, `from vis import ...`)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDGE_DIR = os.path.join(ROOT, "edge")
EDGE_DATA_DIR = os.path.join(EDGE_DIR, "data")
if EDGE_DIR not in sys.path:
    sys.path.insert(0, EDGE_DIR)
