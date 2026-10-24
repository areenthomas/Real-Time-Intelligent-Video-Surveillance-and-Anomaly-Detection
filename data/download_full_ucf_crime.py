"""
UCF-Crime Full Dataset Access & Downloader Utility
Official Reference: "Real-World Anomaly Detection in Surveillance Videos" (Waqas Sultani et al., CVPR 2018)
Center for Research in Computer Vision (CRCV), University of Central Florida
"""

import os
import sys

# Force UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATASET_INFO = {
    "total_videos": 1900,
    "total_hours": 128,
    "anomalous_videos": 950,
    "normal_videos": 950,
    "classes": [
        "Abuse", "Arrest", "Arson", "Assault", "Accident",
        "Burglary", "Explosion", "Fighting", "Robbery", "Shooting",
        "Shoplifting", "Stealing", "Vandalism", "Normal"
    ],
    "official_sources": {
        "Dropbox Split Archive (Author Mirror)": "https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0",
        "CRCV UCF Project Portal": "https://visionlab.uncc.edu/download/summary/60-data/477-ucf-anomaly-detection-dataset",
        "Kaggle Full Video Mirror (110 GB)": "https://www.kaggle.com/datasets/mission-ai/ucf-crime-dataset",
        "Kaggle Pre-Extracted I3D Features (~15 GB)": "https://www.kaggle.com/datasets/odinsam/ucf-crime-i3d-features",
        "Official Sultani GitHub": "https://github.com/WaqasSultani/AnomalyDetectionCVPR2018",
    }
}

def display_access_guide():
    print("=" * 80)
    print("UCF-CRIME: COMPLETE SURVEILLANCE VIDEO DATASET ACCESS GUIDE")
    print("=" * 80)
    print(f"Total Untrimmed CCTV Videos : {DATASET_INFO['total_videos']:,} videos")
    print(f"Total Surveillance Footage  : {DATASET_INFO['total_hours']} Hours (~110 GB Raw MP4)")
    print(f"Monitored Crime Classes     : {len(DATASET_INFO['classes'])} Categories")
    print("-" * 80)
    print("OFFICIAL REPOSITORIES & MIRRORS:")
    for name, url in DATASET_INFO["official_sources"].items():
        print(f"  • {name:<45} : {url}")
    print("=" * 80)
    print("COMMAND LINE FAST-DOWNLOAD (via Kaggle CLI):")
    print("  1. Download Full 110 GB Raw Surveillance Videos:")
    print("     kaggle datasets download -d mission-ai/ucf-crime-dataset -p data/raw_videos/ --unzip")
    print("\n  2. Download Lightweight Precomputed I3D Features (~15 GB, Recommended for Training):")
    print("     kaggle datasets download -d odinsam/ucf-crime-i3d-features -p data/features/i3d_ucf_crime/ --unzip")
    print("=" * 80)

if __name__ == "__main__":
    display_access_guide()

