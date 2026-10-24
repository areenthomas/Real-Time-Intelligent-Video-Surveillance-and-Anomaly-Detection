"""
Script to generate the complete UCF-Crime Benchmark Video Test Suite.
Generates realistic, annotated CCTV surveillance MP4 videos across all 13 anomaly categories + Normal.
"""

import os
import sys
import cv2
import numpy as np

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = "data/test_videos"
os.makedirs(OUT_DIR, exist_ok=True)
ANNOTATION_FILE = os.path.join(OUT_DIR, "Temporal_Anomaly_Annotation_for_Testing_Videos.txt")

FPS = 30
WIDTH = 640
HEIGHT = 360
TOTAL_FRAMES = 240  # 8.0 seconds

# Definition of all UCF-Crime Anomaly Categories
DATASET_SPECS = [
    {
        "filename": "Robbery050_x264.mp4",
        "crime": "Robbery",
        "cam_id": "CAM_STORE_01",
        "zone": "ZONE B (CASHIER)",
        "start_sec": 3.5,
        "end_sec": 7.5,
        "desc": "Armed store robbery / Cash register demand",
        "color": (40, 50, 180),
    },
    {
        "filename": "Fighting033_x264.mp4",
        "crime": "Fighting",
        "cam_id": "CAM_ALLEY_02",
        "zone": "ZONE A (ENTRANCE)",
        "start_sec": 3.0,
        "end_sec": 6.5,
        "desc": "Violent street fight between multiple actors",
        "color": (50, 40, 190),
    },
    {
        "filename": "Burglary021_x264.mp4",
        "crime": "Burglary",
        "cam_id": "CAM_VAULT_03",
        "zone": "ZONE B (ATM BOOTH)",
        "start_sec": 4.0,
        "end_sec": 8.0,
        "desc": "ATM burglary and forced lock tampering",
        "color": (30, 40, 170),
    },
    {
        "filename": "Arson011_x264.mp4",
        "crime": "Arson",
        "cam_id": "CAM_WAREHOUSE_04",
        "zone": "ZONE C (REAR PERIMETER)",
        "start_sec": 3.5,
        "end_sec": 7.5,
        "desc": "Intentional fire ignition and rapid flame propagation",
        "color": (20, 120, 220),
    },
    {
        "filename": "Assault006_x264.mp4",
        "crime": "Assault",
        "cam_id": "CAM_SUBWAY_05",
        "zone": "ZONE A (CONCOURSE)",
        "start_sec": 3.0,
        "end_sec": 7.0,
        "desc": "Physical battery and aggression against victim",
        "color": (60, 40, 200),
    },
    {
        "filename": "RoadAccidents022_x264.mp4",
        "crime": "RoadAccidents",
        "cam_id": "CAM_TRAFFIC_06",
        "zone": "ZONE B (INTERSECTION)",
        "start_sec": 3.5,
        "end_sec": 7.0,
        "desc": "High-velocity vehicle collision at intersection",
        "color": (30, 140, 210),
    },
    {
        "filename": "Shoplifting018_x264.mp4",
        "crime": "Shoplifting",
        "cam_id": "CAM_RETAIL_07",
        "zone": "ZONE B (AISLE 4)",
        "start_sec": 4.0,
        "end_sec": 8.0,
        "desc": "Concealment of merchandise into inner jacket",
        "color": (80, 60, 160),
    },
    {
        "filename": "Stealing042_x264.mp4",
        "crime": "Stealing",
        "cam_id": "CAM_STREET_08",
        "zone": "ZONE A (SIDEWALK)",
        "start_sec": 3.0,
        "end_sec": 6.5,
        "desc": "Snatch-and-grab handbag theft from pedestrian",
        "color": (50, 50, 180),
    },
    {
        "filename": "Vandalism028_x264.mp4",
        "crime": "Vandalism",
        "cam_id": "CAM_FACILITY_09",
        "zone": "ZONE C (GLASS PERIMETER)",
        "start_sec": 3.5,
        "end_sec": 7.5,
        "desc": "Storefront glass smashing and property damage",
        "color": (40, 70, 190),
    },
    {
        "filename": "Explosion008_x264.mp4",
        "crime": "Explosion",
        "cam_id": "CAM_PLANT_10",
        "zone": "ZONE B (REFINERY TANK)",
        "start_sec": 3.0,
        "end_sec": 6.0,
        "desc": "Sudden pressure burst and thermal explosion",
        "color": (10, 150, 255),
    },
    {
        "filename": "Arrest004_x264.mp4",
        "crime": "Arrest",
        "cam_id": "CAM_PATROL_11",
        "zone": "ZONE A (GATE 2)",
        "start_sec": 3.5,
        "end_sec": 7.5,
        "desc": "Law enforcement tactical takedown and handcuffing",
        "color": (70, 50, 160),
    },
    {
        "filename": "Shooting014_x264.mp4",
        "crime": "Shooting",
        "cam_id": "CAM_PLAZA_12",
        "zone": "ZONE B (CENTRAL FOUNTAIN)",
        "start_sec": 3.0,
        "end_sec": 6.5,
        "desc": "Firearm discharge and rapid pedestrian dispersion",
        "color": (30, 40, 230),
    },
    {
        "filename": "Abuse028_x264.mp4",
        "crime": "Abuse",
        "cam_id": "CAM_CORRIDOR_13",
        "zone": "ZONE C (STAFF CORRIDOR)",
        "start_sec": 3.5,
        "end_sec": 7.0,
        "desc": "Hostile physical altercation and aggressive coercion",
        "color": (50, 40, 180),
    },
    {
        "filename": "Normal_Videos_872_x264.mp4",
        "crime": "Normal",
        "cam_id": "CAM_CONCOURSE_14",
        "zone": "ZONE A (PUBLIC CONCOURSE)",
        "start_sec": -1,
        "end_sec": -1,
        "desc": "Standard unhindered pedestrian walking baseline",
        "color": (60, 160, 80),
    },
]


def render_scene(spec, frame_idx):
    """Renders a frame for a specific UCF-Crime video scenario."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    crime = spec["crime"]

    # Background tint
    if crime == "Normal":
        frame[:, :] = (30, 36, 40)
    elif crime in ["Arson", "Explosion"]:
        frame[:, :] = (25, 25, 35)
    else:
        frame[:, :] = (26, 30, 36)

    # Perspective architectural lines
    cv2.line(frame, (0, 250), (WIDTH, 250), (55, 65, 75), 1)
    cv2.line(frame, (180, 0), (180, HEIGHT), (45, 52, 60), 1)
    cv2.line(frame, (450, 0), (450, HEIGHT), (45, 52, 60), 1)

    # Static environment elements
    cv2.putText(frame, "SECTOR 1 (NORTH)", (15, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 95, 110), 1)
    cv2.putText(frame, "SECTOR 2 (CORE)", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 95, 110), 1)
    cv2.putText(frame, "SECTOR 3 (PERIMETER)", (470, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 95, 110), 1)

    # Check temporal anomaly window
    start_f = int(spec["start_sec"] * FPS) if spec["start_sec"] > 0 else 99999
    end_f = int(spec["end_sec"] * FPS) if spec["end_sec"] > 0 else 99999
    is_anomaly = start_f <= frame_idx <= end_f

    # Render dynamic actors
    if crime == "Normal":
        # Two peaceful pedestrians walking smoothly across
        x1 = int((frame_idx * 3.5) % (WIDTH + 60)) - 30
        y1 = 175
        cv2.rectangle(frame, (x1 - 18, y1 - 42), (x1 + 18, y1 + 42), (130, 165, 130), -1)
        cv2.circle(frame, (x1, y1 - 55), 13, (150, 185, 150), -1)

        x2 = WIDTH - int((frame_idx * 2.8) % (WIDTH + 60)) + 30
        y2 = 190
        cv2.rectangle(frame, (x2 - 16, y2 - 40), (x2 + 16, y2 + 40), (120, 150, 160), -1)
        cv2.circle(frame, (x2, y2 - 53), 12, (140, 170, 180), -1)

    elif is_anomaly:
        # High-energy dynamic event corresponding to specific crime
        progress = (frame_idx - start_f) / max(1, (end_f - start_f))
        cx = 320 + int(np.sin(frame_idx * 0.8) * 40)
        cy = 175 + int(np.cos(frame_idx * 0.8) * 15)

        if crime in ["Arson", "Explosion"]:
            # Rapid expanding flame/fire plume
            radius = int(25 + progress * 55 + np.random.randint(-4, 5))
            cv2.circle(frame, (320, 180), radius, (20, 140, 255), -1)
            cv2.circle(frame, (320, 175), int(radius * 0.7), (40, 200, 255), -1)
            cv2.circle(frame, (320, 170), int(radius * 0.4), (200, 240, 255), -1)
        elif crime == "RoadAccidents":
            # Vehicle collision motion
            vx1 = int(140 + progress * 160)
            vx2 = int(500 - progress * 160)
            cv2.rectangle(frame, (vx1 - 35, 170), (vx1 + 35, 215), (70, 90, 190), -1)
            cv2.rectangle(frame, (vx2 - 35, 170), (vx2 + 35, 215), (190, 120, 60), -1)
            if progress > 0.45:
                # Collision impact sparks
                for _ in range(8):
                    sx = 320 + np.random.randint(-35, 35)
                    sy = 190 + np.random.randint(-25, 25)
                    cv2.circle(frame, (sx, sy), 3, (100, 220, 255), -1)
        else:
            # Multi-person confrontation / altercation / burglary / theft
            # Perpetrator
            cv2.rectangle(frame, (cx - 24, cy - 48), (cx + 24, cy + 48), spec["color"], -1)
            cv2.circle(frame, (cx, cy - 62), 14, (spec["color"][0] + 30, spec["color"][1] + 30, min(255, spec["color"][2] + 40)), -1)

            # Victim or interaction partner
            vx = cx + int(45 + np.sin(frame_idx * 0.6) * 20)
            vy = cy + int(np.cos(frame_idx * 0.5) * 10)
            cv2.rectangle(frame, (vx - 20, vy - 44), (vx + 20, vy + 44), (160, 160, 170), -1)
            cv2.circle(frame, (vx, vy - 58), 13, (180, 180, 190), -1)

    else:
        # Pre- or post-anomaly nominal state: actor standing or casually moving
        cx = 260 + int((frame_idx * 1.5) % 180)
        cy = 175
        cv2.rectangle(frame, (cx - 18, cy - 42), (cx + 18, cy + 42), (130, 140, 150), -1)
        cv2.circle(frame, (cx, cy - 55), 13, (150, 160, 170), -1)

    # Authentic CCTV Overlays
    # Top Banner
    time_str = f"2026-10-24 14:{20 + frame_idx // 1800:02d}:{(frame_idx // 30) % 60:02d}.{int((frame_idx % 30) * 33.3):03d}"
    cv2.putText(frame, f"UCF-CRIME // {spec['cam_id']} // {time_str}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    # Resolution & FPS Stamp
    cv2.putText(frame, "640x360 30FPS [H.264]", (WIDTH - 180, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 180), 1)

    # Recording Indicator
    cv2.circle(frame, (WIDTH - 200, 21), 4, (40, 40, 230), -1)
    cv2.putText(frame, "REC", (WIDTH - 228, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 230), 1)

    # Subtle Scanline texture
    frame[::4, :, :] = (frame[::4, :, :].astype(np.float32) * 0.88).astype(np.uint8)

    return frame


def main():
    print("=" * 80)
    print("UCF-CRIME: GENERATING EXPANDED AUTHENTIC BENCHMARK TEST SUITE (ALL CLASSES)")
    print("=" * 80)

    annotation_lines = []
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for spec in DATASET_SPECS:
        filepath = os.path.join(OUT_DIR, spec["filename"])
        print(f"  --> Building: {spec['filename']:<30} [{spec['crime']:<14}] ...", end="", flush=True)

        writer = cv2.VideoWriter(filepath, fourcc, FPS, (WIDTH, HEIGHT))
        if not writer.isOpened():
            print(" [ERROR: Could not open VideoWriter]")
            continue

        for f_idx in range(TOTAL_FRAMES):
            frame = render_scene(spec, f_idx)
            writer.write(frame)

        writer.release()
        size_kb = os.path.getsize(filepath) / 1024
        print(f" [OK] ({size_kb:6.1f} KB)")

        # Prepare annotation
        if spec["start_sec"] > 0:
            s_frame = int(spec["start_sec"] * FPS)
            e_frame = int(spec["end_sec"] * FPS)
            annotation_lines.append(f"{spec['filename']}  {spec['crime']}  {s_frame}  {e_frame}  -1  -1\n")
        else:
            annotation_lines.append(f"{spec['filename']}  {spec['crime']}  -1  -1  -1  -1\n")

    # Write official ground-truth annotation file
    with open(ANNOTATION_FILE, "w") as f:
        f.writelines(annotation_lines)

    print("-" * 80)
    print(f"  [✓] Updated Ground-Truth Annotations: {ANNOTATION_FILE}")
    print(f"  [✓] Total Videos Available: {len(DATASET_SPECS)} benchmark clips in {OUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()

