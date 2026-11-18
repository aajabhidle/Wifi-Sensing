# CSI-Based Human Pose Reconstruction & Gesture Recognition with Spiking Neural Networks

A privacy-preserving, camera-free gesture recognition pipeline using Wi-Fi **Channel State Information (CSI)**. The system reconstructs 2D human silhouettes and 17-body COCO keypoints directly from Wi-Fi CSI using a deep neural network, and subsequently classifies dynamic gestures (Both Hands Raise, Left Hand Raise, Right Hand Raise) using an energy-efficient **Spiking Neural Network (SNN)** equipped with Leaky Integrate-and-Fire (LIF) neurons.

---

## Architecture Overview

```
                      Wi-Fi Packets (CSI)
                              │
                              ▼
                  [ CSI Preprocessing & Sync ]
                (PicoScenes / Phase Calibration)
                              │
                              ▼
                 [ SiWiS Pose Reconstruction ]
          (Predicts 2D Body Silhouette & 17 Keypoints)
                              │
                              ▼
            [ Spatial-Temporal Feature Extraction ]
             - 192-d average-pooled body mask
             - 36-d keypoint & speed features
                              │
                              ▼
              [ Spiking Neural Network (SNN) ]
               - Leaky Integrate-and-Fire (LIF)
               - Temporal surrogate gradients
                              │
                              ▼
           [ Dynamic Gesture Classification ]
                 (BHR / LHR / RHR)
```

---

## Repository Structure

```
percom/
├── README.md                      # Project documentation and benchmark results
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git ignore rules for datasets, binaries, checkpoints
│
├── models/                        # Neural network architectures
│   ├── __init__.py
│   └── pose_snn.py                # Spiking Neural Network (SNN) with LIF neurons & surrogate gradient
│
├── pipelines/                     # End-to-end training and evaluation protocols
│   ├── __init__.py
│   ├── run_protocols_5p.py        # Master runner for Protocols A, B, and C across 5 participants
│   ├── scenario2_pooled_pipeline.py # Protocol B: Pooled 5-participant training and cross-validation
│   ├── scenario3_lopo_pipeline.py   # Protocol C: Leave-One-Person-Out (uncalibrated generalization)
│   ├── run_step3_resegment.py     # Video/CSI boundary resegmentation pipeline
│   ├── train_percom_v2.py         # SiWiS pose reconstruction training and evaluation
│   ├── train_snn.py               # Standalone SNN training loop with cross-entropy loss
│   └── bisect_snn.py              # SNN bisection and hyperparameter diagnostics
│
├── preprocessing/                 # CSI extraction, synchronization, and labeling
│   ├── __init__.py
│   ├── csi_to_npz.py              # PicoScenes .csi to NumPy (.npz) converter
│   ├── csi_video_sync.py          # Millisecond-accurate CSI packet to video frame alignment
│   ├── preprocess_csi.py          # Phase unwrapping, Hampel filtering, Doppler shifts
│   ├── make_dataset.py            # Segment-level gesture dataset creation
│   ├── video_segments.py          # Peak detection and boundary segmentation
│   ├── mp_cut_and_run.py          # MediaPipe hand landmark extraction
│   ├── hand_state.py              # Hand elevation classifier
│   ├── pose_video.py              # Visualizer rendering CSI-predicted poses to video (.mp4)
│   ├── convert_vardaan_csi.py     # CSI format conversion utility
│   ├── prepare_vardaan.py         # Participant data formatting pipeline
│   ├── setup_vardaan_folders.py   # Participant directory structure initialization
│   └── standardize_vardaan.py     # Naming standardization for trial recordings
│
├── evaluation/                    # Metrics, auditing, and confusion matrix plotting
│   ├── __init__.py
│   ├── audit_summary_table.py     # Audit script computing Macro F1, Precision, and Recall
│   ├── audit_vardaan_step1.py     # Synchronization and label consistency verification
│   ├── calc_both_metrics.py       # Metrics calculator for multimodal features
│   ├── make_scenario2_reports.py  # Markdown and JSON report generator for Protocol B
│   ├── report_5p.py               # 5-participant LaTeX and Markdown summary reporting
│   ├── verify_5p_dataset.py       # Integrity and completeness verifier for trial data
│   ├── verify_cm_vs_table.py      # Cross-validation between prediction tables and CM plots
│   ├── regen_all_cm.py            # High-resolution confusion matrix generation (300 DPI)
│   └── replot_cm.py               # Single confusion matrix plotting utility
│
├── scripts/                       # Shell scripts for hardware and environment execution
│   ├── record.sh                  # Automated CSI + webcam simultaneous capture script
│   ├── csi_check.sh               # Real-time packet rate and health check for CSI streams
│   ├── cm.sh                      # Batch confusion matrix execution
│   └── quick_notes.txt            # Operator environment commands
│
└── docs/                          # Presentation decks, manuals, and architectural diagrams
    ├── figures/
    │   ├── project_flowchart.png  # Complete architecture flowchart
    │   ├── slide19_original.png   # Benchmark reference slide
    │   └── slide20_original.png   # Benchmark reference slide
    ├── manuals/
    │   ├── GPU_Server_Usage_Policy_v2-HCD-Final.pdf
    │   └── GPU_Server_User_Manual-HCD.pdf
    └── presentation/
        ├── btp_updated.pptx       # Updated project presentation slide deck
        ├── update_btp_pptx.py     # Automated script to inject tables and figures into slides
        └── inspect_slide.py       # Slide XML inspection utility
```

---

## Experimental Benchmark Results (5 Participants)

The system was evaluated on a comprehensive 5-participant dataset (Priyanshu, Sanchit, Sir, Vansh, Vardaan) across three distinct operational protocols:

### Summary of Protocols

| Protocol | Evaluation Paradigm | Accuracy (Joints+Spd) | Macro F1 | Accuracy (Both+Spd) |
|---|---|:---:|:---:|:---:|
| **Protocol A** | **Participant-Wise (Individually Calibrated)** | **86.9% ± 4.8%** | **86.6%** | **91.5%** |
| **Protocol B** | **Pooled Calibration (Multi-User Model)** | **81.4% ± 1.8%** | **81.5%** | **87.2%** |
| **Protocol C** | **Leave-One-Person-Out (Uncalibrated)** | **44.1% ± 8.9%** | **40.8%** | **32.4%** |

### Protocol A: Per-Participant Breakdown (Individually Calibrated)

| Participant | Accuracy (Joints+Spd) | Macro F1 | Macro Precision | Macro Recall | Accuracy (Both+Spd) |
|---|:---:|:---:|:---:|:---:|:---:|
| **P1 (Priyanshu)** | 91.5% ± 3.6% | 91.5% | 91.8% | 91.6% | 92.4% |
| **P2 (Sanchit)**   | 85.7% ± 2.9% | 85.1% | 86.4% | 85.5% | 93.0% |
| **P3 (Sir)**       | 81.0% ± 10.5% | 81.1% | 84.0% | 81.2% | 82.4% |
| **P4 (Vansh)**     | 93.4% ± 2.2% | 92.3% | 92.8% | 92.5% | 93.6% |
| **P5 (Vardaan)**   | 83.1% ± 2.2% | 83.1% | 84.4% | 83.5% | 96.0% |
| **MEAN (All 5)**   | **86.9% ± 4.8%** | **86.6%** | **87.9%** | **86.9%** | **91.5%** |

---

## Installation & Setup

### 1. Prerequisites
- Python 3.9+
- PyTorch >= 2.0.0 with CUDA support (for GPU acceleration)

### 2. Install Python Dependencies
```bash
git clone https://github.com/<your-username>/percom.git
cd percom
pip install -r requirements.txt
```

---

## Quick Start & Usage

### 1. Run Benchmark Protocols
To run all three evaluation protocols across the 5 participants:
```bash
python pipelines/run_protocols_5p.py
```

To run individual protocols:
```bash
# Protocol B: Pooled Calibration
python pipelines/scenario2_pooled_pipeline.py

# Protocol C: Leave-One-Person-Out (LOPO)
python pipelines/scenario3_lopo_pipeline.py
```

### 2. Train the Spiking Neural Network (SNN)
```bash
python pipelines/train_snn.py --data_dir <path_to_poses_npz> --epochs 100 --lr 0.001
```

### 3. Generate Confusion Matrices & Verification Tables
```bash
# Verify master predictions and generate normalized confusion matrices
python evaluation/verify_cm_vs_table.py
python evaluation/regen_all_cm.py
```

---

## License & Citation
Developed as part of the PerCom Research Project at IIIT Delhi.
