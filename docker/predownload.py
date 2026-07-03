"""Bake face-model weights into the image at build time (best-effort).

Failure is non-fatal: if the build machine is offline the service still works
with whichever backend's models ARE present, or degrades to engine-disabled
(API up, enrollment 503) with a clear log message.
"""
import os
import sys
import urllib.request

ROOT = os.environ.get("FACEWEIGHTS", "/opt/faceweights")

# OpenCV zoo models are stored in Git LFS — media.githubusercontent.com serves
# the actual binaries (raw.githubusercontent.com would return LFS pointers).
OPENCV_MODELS = {
    "yunet.onnx": ("https://media.githubusercontent.com/media/opencv/opencv_zoo/"
                   "main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    "sface.onnx": ("https://media.githubusercontent.com/media/opencv/opencv_zoo/"
                   "main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"),
}


def fetch_opencv() -> None:
    d = os.path.join(ROOT, "opencv")
    os.makedirs(d, exist_ok=True)
    for name, url in OPENCV_MODELS.items():
        dst = os.path.join(d, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 10_000:
            print(f"  {name}: cached")
            continue
        try:
            urllib.request.urlretrieve(url, dst)
            size = os.path.getsize(dst)
            if size < 10_000:            # LFS pointer / error page
                os.unlink(dst)
                raise RuntimeError(f"suspicious size {size}")
            print(f"  {name}: {size/1e6:.1f} MB")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: FAILED ({exc})")


def fetch_insightface() -> None:
    try:
        from insightface.app import FaceAnalysis
        FaceAnalysis(name="buffalo_l", root=ROOT,
                     providers=["CPUExecutionProvider"],
                     allowed_modules=["detection", "recognition"])
        print("  buffalo_l: cached OK")
    except Exception as exc:  # noqa: BLE001
        print(f"  buffalo_l: FAILED ({exc})")


if __name__ == "__main__":
    print("Pre-downloading OpenCV zoo face models...")
    fetch_opencv()
    print("Pre-downloading InsightFace buffalo_l...")
    fetch_insightface()
    sys.exit(0)     # never fail the build
