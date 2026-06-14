"""
Dataset collection script for FaceVision AI experiments.

Usage:
    python experiments/collect_dataset.py --person person_01 --condition frontal
    python experiments/collect_dataset.py --person person_01 --condition pose15
    python experiments/collect_dataset.py --person person_01 --condition pose30
    python experiments/collect_dataset.py --person person_01 --condition dark
    python experiments/collect_dataset.py --person person_01 --condition occluded

Controls during capture:
    SPACE  — capture frame (saves to dataset folder)
    Q      — quit session

Each session captures 5 images per condition. Progress is shown on screen.
After all persons done, metadata.csv is auto-generated.
"""

import argparse
import csv
import os
import time

import cv2

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
META_FILE   = os.path.join(DATASET_DIR, "metadata.csv")

CONDITIONS = {
    "frontal":  {"pose": "0deg",  "lighting": "even",   "occlusion": "none"},
    "pose15":   {"pose": "15deg", "lighting": "even",   "occlusion": "none"},
    "pose30":   {"pose": "30deg", "lighting": "even",   "occlusion": "none"},
    "dark":     {"pose": "0deg",  "lighting": "low",    "occlusion": "none"},
    "occluded": {"pose": "0deg",  "lighting": "even",   "occlusion": "mask"},
}

INSTRUCTIONS = {
    "frontal":  "Look straight at camera, normal lighting",
    "pose15":   "Turn head ~15 degrees to the right",
    "pose30":   "Turn head ~30 degrees to the right",
    "dark":     "Reduce lighting (turn off lights / cover camera partially)",
    "occluded": "Wear a mask or cover lower face with hand",
}

TARGET_COUNT = 5


def collect(person_id: str, condition: str, camera_idx: int = 0):
    if condition not in CONDITIONS:
        print(f"Unknown condition: {condition}. Choose from: {list(CONDITIONS)}")
        return

    person_dir = os.path.join(DATASET_DIR, person_id)
    os.makedirs(person_dir, exist_ok=True)

    meta = CONDITIONS[condition]
    instr = INSTRUCTIONS[condition]
    print(f"\n=== Collecting: {person_id} / {condition} ===")
    print(f"Instructions: {instr}")
    print(f"Press SPACE to capture (need {TARGET_COUNT} images), Q to quit\n")

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    time.sleep(1)
    saved = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        count = len(saved)
        overlay = frame.copy()
        label = f"{person_id} | {condition} | {count}/{TARGET_COUNT}"
        cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.putText(overlay, instr, (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 0), 1)
        cv2.putText(overlay, "SPACE=capture  Q=quit", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Dataset Collection", overlay)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            idx = len(saved) + 1
            fname = f"{condition}_{idx:03d}.jpg"
            fpath = os.path.join(person_dir, fname)
            cv2.imwrite(fpath, frame)
            saved.append(fname)
            print(f"  Saved {fname} ({count+1}/{TARGET_COUNT})")

            if len(saved) >= TARGET_COUNT:
                print(f"\nDone! {TARGET_COUNT} images captured.")
                break

    cap.release()
    cv2.destroyAllWindows()

    # Append to metadata.csv
    existing = set()
    if os.path.exists(META_FILE):
        with open(META_FILE, newline="") as f:
            for row in csv.DictReader(f):
                existing.add(row["image_file"])

    new_rows = []
    for fname in saved:
        if fname not in existing:
            new_rows.append({
                "person_id": person_id,
                "image_file": fname,
                "session": "1",
                "pose": meta["pose"],
                "lighting": meta["lighting"],
                "occlusion": meta["occlusion"],
                "camera": "webcam",
                "resolution": "640x480",
                "note": "",
            })

    if new_rows:
        write_header = not os.path.exists(META_FILE) or os.path.getsize(META_FILE) < 10
        with open(META_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "person_id", "image_file", "session",
                "pose", "lighting", "occlusion", "camera", "resolution", "note"
            ])
            if write_header:
                writer.writeheader()
            writer.writerows(new_rows)
        print(f"Appended {len(new_rows)} rows to metadata.csv")

    return saved


def summarize():
    """Print dataset summary."""
    if not os.path.exists(META_FILE):
        print("No metadata.csv found.")
        return

    from collections import defaultdict
    counts = defaultdict(lambda: defaultdict(int))
    total = 0

    with open(META_FILE, newline="") as f:
        for row in csv.DictReader(f):
            pid = row["person_id"]
            cond_key = f"{row['pose']}_{row['lighting']}_{row['occlusion']}"
            counts[pid][cond_key] += 1
            total += 1

    print(f"\n=== Dataset Summary: {total} images, {len(counts)} persons ===")
    for pid in sorted(counts):
        print(f"  {pid}: {dict(counts[pid])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect face dataset for experiments")
    parser.add_argument("--person", required=True, help="e.g. person_01")
    parser.add_argument("--condition", required=True,
                        choices=list(CONDITIONS), help="Capture condition")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--summary", action="store_true", help="Show dataset summary only")
    args = parser.parse_args()

    if args.summary:
        summarize()
    else:
        collect(args.person, args.condition, args.camera)
        summarize()
