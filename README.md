# Arrhythmia Detection Web Application 🫀

**Early Detection of Arrhythmia using Deep Spiking Neural Network (DSNN)**

This application is a full-stack medical AI solution for classifying cardiac arrhythmias from ECG recordings. It features a robust deep learning pipeline, real-time training visualization, and a clinical-grade analysis dashboard.

---

## 📂 Project Structure

```text
Arrhythmia-Detection/
├── backend/                   # Flask REST API & ML Services
│   ├── models/                # Trained PyTorch models (.pth)
│   ├── routes/                # API Endpoints (api.py, web.py)
│   ├── services/              # Core Logic
│   │   ├── train_dsnn.py      # DSNN Architecture & Training Pipeline
│   │   └── converter.py       # MIT-BIH to EDF/QRS Conversion
│   ├── results/               # Persisted Analysis Reports (JSON)
│   ├── images/                # Generated Training Visualizations (Plots)
│   └── app.py                 # Flask Entry Point
├── frontend/                  # React + Vite Frontend
│   ├── src/
│   │   ├── components/        # Reusable UI (Charts, Modals, Layouts)
│   │   ├── pages/             # Dashboard, Training, Analysis, Settings
│   │   └── App.jsx            # Routing & Global State
│   └── vite.config.js
└── Dataset/                   # ECG Data Repository
    ├── MIT-BIH/               # 40 Gold-Standard training records (.edf, .qrs)
    └── test/                  # 8 Unseen evaluation records
```

---

## ⚙️ Core Implementation Pipeline

The application follows a rigorous 3-stage pipeline to transform raw ECG data into clinical insights.

### 1. Data Conversion & Pre-processing
- **Source**: MIT-BIH Arrhythmia Database.
- **Conversion**: Raw `.dat`/`.hea` files are converted to `.edf` (European Data Format) for clinical compatibility.
- **Filtering**: A **0.5Hz - 40Hz Butterworth Bandpass Filter** is applied to remove baseline wander (breathing) and powerline interference.
- **Normalization**: Per-segment **Z-Score Normalization** ensures the model learns heartbeat **morphology** (shape) rather than absolute voltage amplitudes.

### 2. Deep Spiking Neural Network (DSNN) Training
- **Architecture**: A Deep Convolutional SNN optimized for temporal signal processing.
- **Segmentation**: 
    - **Peak-Triggered**: Segments are centered exactly on the R-peak (detected via `.qrs` annotations).
    - **Optimization**: Uses **Focal Loss** to combat class imbalance, forcing the model to prioritize rare arrhythmias over common "Normal" beats.
- **Tracking**: Real-time progress is broadcast to the frontend via status polling, displaying live Loss/Accuracy curves.

### 3. Clinical Analysis (Inference)
- **High-Confidence Diagnosis**: The system analyzes full ECG recordings and provides a primary diagnosis with a confidence score.
- **Peer-Review Ready**: Detailed breakdown of every segment type (Normal, AFib, Ventricular, etc.) is provided.
- **Safety Thresholds**: Results falling below the configurable threshold (default 60%) are flagged as **Inconclusive** to ensure clinical safety.

---

## 📊 Arrhythmia Classes
The model classifies ECG signals into 6 distinct categories:
1. **Normal Sinus Rhythm**
2. **Atrial Fibrillation (AFib)**
3. **Ventricular Arrhythmia**
4. **Conduction Block**
5. **Premature Contraction**
6. **ST Segment Abnormality**

---

## 🛠️ Getting Started

### Prerequisites
- **Python 3.10+**
- **Node.js 18+**
- **CUDA** (Optional, for GPU-accelerated training)

### Setup & Installation

1. **Install Backend Dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Install Frontend Dependencies**:
   ```bash
   cd frontend
   npm install
   ```

3. **Run the Application**:
   Open two terminals:
   - **Terminal 1 (Backend)**: `cd backend && python app.py`
   - **Terminal 2 (Frontend)**: `cd frontend && npm run dev`

---

## 📝 Tech Stack
- **Frontend**: React, Vite, Recharts, Lucide React, Axios.
- **Backend**: Flask, PyTorch (Deep Learning), WFDB (Bio-signal Processing), SciPy (Signal Processing), NumPy.
- **Dataset**: MIT-BIH Arrhythmia Database (PhysioNet).

---
*Developed for accurate, early detection of cardiac conditions.*
