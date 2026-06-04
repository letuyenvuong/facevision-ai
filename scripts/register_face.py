"""CLI script to register a face from terminal.

Usage:
    python scripts/register_face.py --name "Alice" --image path/to/photo.jpg
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Register a face into FaceVision AI database")
    parser.add_argument("--name",  required=True, help="Person's name")
    parser.add_argument("--image", required=True, help="Path to the face image")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    from modules.recognition import FaceRecognizer
    recognizer = FaceRecognizer()
    try:
        filepath = recognizer.register(args.name, args.image)
        print(f"[OK] Registered '{args.name}' — embedding saved to {filepath}")
    except Exception as e:
        print(f"[ERROR] Registration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
