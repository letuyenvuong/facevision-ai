"""Run this script to verify all dependencies are installed correctly."""
import sys


def check(name, import_fn):
    try:
        result = import_fn()
        version = getattr(result, "__version__", "?")
        print(f"  [OK] {name:<20} {version}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name:<20} {e}")
        return False


def main():
    print("\n=== FaceVision AI — Environment Check ===\n")
    results = []

    results.append(check("Python", lambda: type(sys)()))

    import importlib
    deps = [
        ("flask",         "flask"),
        ("opencv-python", "cv2"),
        ("numpy",         "numpy"),
        ("Pillow",        "PIL"),
        ("deepface",      "deepface"),
        ("mtcnn",         "mtcnn"),
        ("scikit-learn",  "sklearn"),
        ("scipy",         "scipy"),
        ("tensorflow",    "tensorflow"),
    ]
    for label, mod in deps:
        results.append(check(label, lambda m=mod: importlib.import_module(m)))

    print("\n--- Webcam access ---")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        opened = cap.isOpened()
        cap.release()
        status = "OK" if opened else "NOT FOUND (no webcam or index wrong)"
        print(f"  [{'OK' if opened else 'WARN'}] Webcam (index 0): {status}")
    except Exception as e:
        print(f"  [FAIL] Webcam check: {e}")

    passed = sum(results)
    total = len(results)
    print(f"\n=== {passed}/{total} checks passed ===\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
