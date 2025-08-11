import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `import pmm` works when running plain `pytest`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Default to fast test mode unless explicitly disabled
os.environ.setdefault("PMM_TEST_MODE", "1")
