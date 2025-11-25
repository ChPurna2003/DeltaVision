import cv2
import numpy as np
import os
from pathlib import Path

INPUT_DIR = "input"
OUTPUT_DIR = "output"

# Tuned offsets for your dataset
X_SHIFT = -12     # move thermal left
Y_SHIFT = 18      # move thermal down


def extract_key(name):
    n = name.upper()
    if n.endswith("_T.JPG"):
        return n[:-6]
    if n.endswith("_Z.JPG"):
        return n[:-6]
    return None


def get_pairs():
    files = sorted(os.listdir(INPUT_DIR))
    pairs = {}

    for f in files:
        key = extract_key(f)
        if not key:
            continue

        if f.upper().endswith("_T.JPG"):
            pairs.setdefault(key, {})["thermal"] = f
        elif f.upper().endswith("_Z.JPG"):
            pairs.setdefault(key, {})["rgb"] = f

    # return only complete pairs
    return {k: v for k, v in pairs.items() if "thermal" in v and "rgb" in v}


def align_simple(rgb, thermal):
    h, w = rgb.shape[:2]
    h2, w2 = thermal.shape[:2]

    # scale thermal to match RGB height
    scale = h / h2
    new_w = int(w2 * scale)
    thermal_resized = cv2.resize(thermal, (new_w, h))

    # create blank canvas
    canvas = np.zeros_like(rgb)

    # horizontally center + apply offset
    x0 = (w - new_w) // 2 + X_SHIFT
    y0 = Y_SHIFT

    # clamp to avoid out of bounds
    x0 = max(0, min(x0, w - new_w))
    y0 = max(0, min(y0, h - h))

    # place resized thermal on canvas
    canvas[y0:y0+h, x0:x0+new_w] = thermal_resized[:, :new_w]

    return canvas


def process():
    # create folders
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    Path(f"{OUTPUT_DIR}/rgb").mkdir(exist_ok=True)
    Path(f"{OUTPUT_DIR}/thermal").mkdir(exist_ok=True)

    pairs = get_pairs()
    print(f"Found {len(pairs)} valid pairs.\n")

    for key, p in pairs.items():
        print(f"Processing {key}...")

        rgb_path = f"{INPUT_DIR}/{p['rgb']}"
        thr_path = f"{INPUT_DIR}/{p['thermal']}"

        rgb = cv2.imread(rgb_path)
        thermal = cv2.imread(thr_path)

        aligned = align_simple(rgb, thermal)

        # save separately
        out_rgb = f"{OUTPUT_DIR}/rgb/{key}_rgb.jpg"
        out_thr = f"{OUTPUT_DIR}/thermal/{key}_aligned.jpg"

        cv2.imwrite(out_rgb, rgb)
        cv2.imwrite(out_thr, aligned)

        print(" ✔ Saved RGB →", out_rgb)
        print(" ✔ Saved Thermal-Aligned →", out_thr)

    print("\nDone! All images saved in separate folders.")


if __name__ == "__main__":
    process()
