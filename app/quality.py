"""
app/quality.py — Face-quality gates.

Two tiers, per the accuracy plan:

  Enrollment (strict)  — a bad reference photo is the #1 false-positive source,
                         so uploads are REJECTED with a human-readable reason
                         unless they contain exactly one sharp, large face.
  Runtime (permissive) — camera frames only need to be good enough to score;
                         tiny/blurred faces are silently skipped so they can
                         never produce a match at all.

Blur is measured as the variance of the Laplacian on the grayscale face crop —
cheap and a reliable proxy for focus/motion blur.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from settings import settings
from app.engine import DetectedFace


@dataclass
class QualityResult:
    ok: bool
    reason: str = ""
    face: Optional[DetectedFace] = None
    blur: float = 0.0


def face_crop(img_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = face.bbox
    xi1, yi1 = max(0, int(x1)), max(0, int(y1))
    xi2, yi2 = min(w, int(x2)), min(h, int(y2))
    if xi2 <= xi1 or yi2 <= yi1:
        return np.empty((0, 0, 3), np.uint8)
    return img_bgr[yi1:yi2, xi1:xi2]


def blur_score(crop_bgr: np.ndarray) -> float:
    if crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _min_side(face: DetectedFace) -> float:
    x1, y1, x2, y2 = face.bbox
    return min(x2 - x1, y2 - y1)


def check_enrollment(img_bgr: np.ndarray, faces: list[DetectedFace]) -> QualityResult:
    """Strict gate for reference-photo uploads. Returns the single good face."""
    if not faces:
        return QualityResult(False, "no face detected in the photo")
    if len(faces) > 1:
        return QualityResult(
            False, f"{len(faces)} faces detected — upload a photo with exactly one person")
    f = faces[0]
    if f.score < settings.enroll_min_det_score:
        return QualityResult(
            False, f"face detection confidence too low ({f.score:.2f} < "
                   f"{settings.enroll_min_det_score}) — use a clearer photo")
    side = _min_side(f)
    if side < settings.enroll_min_face_px:
        return QualityResult(
            False, f"face too small ({side:.0f}px < {settings.enroll_min_face_px}px) "
                   f"— upload a closer/higher-resolution photo")
    blur = blur_score(face_crop(img_bgr, f))
    if blur < settings.enroll_min_blur:
        return QualityResult(
            False, f"photo too blurry (sharpness {blur:.0f} < "
                   f"{settings.enroll_min_blur}) — upload a sharper photo", blur=blur)
    if f.embedding is None:
        return QualityResult(False, "could not compute a face embedding — try another photo")
    return QualityResult(True, face=f, blur=blur)


def check_runtime(img_bgr: np.ndarray, face: DetectedFace) -> bool:
    """Permissive gate for live camera frames — skip unusable faces silently."""
    if face.embedding is None:
        return False
    if face.score < settings.runtime_min_det_score:
        return False
    if _min_side(face) < settings.runtime_min_face_px:
        return False
    if blur_score(face_crop(img_bgr, face)) < settings.runtime_min_blur:
        return False
    return True
