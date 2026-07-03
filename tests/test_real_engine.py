"""Smoke test against the REAL face backend (skips when no models baked).

Runs in a SUBPROCESS: creating an onnxruntime session inside a pytest process
that has already loaded cv2/fastapi/scipy segfaults due to a native-library
state collision. A fresh interpreter is also exactly how the engine runs in
production (loaded once at service startup), so this is the honest test shape.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parent.parent)

_PROBE = r"""
import json, sys
import numpy as np
from app.engine import build_engine
eng = build_engine()
if eng is None:
    print(json.dumps({"available": False}))
    sys.exit(0)
faces = eng.analyze(np.zeros((480, 640, 3), np.uint8))
print(json.dumps({
    "available": True,
    "backend": eng.name,
    "device": eng.device,
    "dim": eng.dim,
    "threshold": eng.match_threshold,
    "faces_on_empty": len(faces),
}))
"""


def test_real_engine_loads_and_analyzes():
    proc = subprocess.run([sys.executable, "-c", _PROBE],
                          capture_output=True, text=True, timeout=120,
                          cwd=_ROOT)
    assert proc.returncode == 0, f"engine probe crashed:\n{proc.stderr[-2000:]}"
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    if not result["available"]:
        pytest.skip("no real face backend available in this environment")
    assert result["backend"] in ("insightface", "opencv")
    assert result["dim"] in (128, 512)
    assert 0.0 < result["threshold"] < 1.0
    assert result["device"] in ("cpu", "cuda")
    assert result["faces_on_empty"] == 0          # no hallucinated faces
