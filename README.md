# Arrhythmia Detection Web Application 🫀

**A comprehensive full-stack medical AI solution for the early detection and classification of cardiac arrhythmias using a Deep Spiking Neural Network (DSNN).**

This application aims to provide clinical-grade analysis of ECG recordings by leveraging a robust deep learning pipeline while ensuring high confidence and detailed breakdown of cardiac conditions. It features real-time model training visualization, an interactive clinical dashboard, and a seamless ECG reader integration.

---

## 🚀 Project Understanding & Features

This project was developed to bridge the gap between advanced deep learning techniques in bio-signal processing and an accessible, user-friendly clinical interface.

### Key Features
- **ECG Reader & Uploads**: Flexible mode selection for file uploads (`.dat`/`.hea`, `.edf`, or `.qrs`), complete with robust background processing.
- **Dynamic Visualization & Downloads**: Real-time rendering of ECG signals via interactive visualizations. Generated charts utilize image caching for optimized performance and are programmatically downloadable for clinical records.
- **Deep Learning Pipeline**: Hosted as a **Dual-Mode Architecture**. Currently features an isolated Deep Spiking Neural Network (DSNN) pipeline handling raw temporal waveforms, and an independent ultra-shallow Residual MLP engineered exclusively for tabular `.csv` morphology features to manage dense covariate shifts.
- **Live Training Dashboard**: Real-time progress broadcasts via status polling to the frontend for both RAW and CSV pipelines concurrently. Live loss and accuracy curves allow researchers and clinicians to monitor model performance dynamically without cross-contamination.
- **Historical Analysis**: Maintain a history of uploaded datasets and generated prediction reports for future comparison.
- **Configurable Settings**: A dedicated settings panel allows users to configure confidence thresholds; results dropping below these logic checks are safely flagged as **Inconclusive**.

---

## 🧠 Core Implementation & Architecture

The application implements a strict 3-stage clinical pipeline converting raw signals to intelligent insights.

### 1. Data Processing
- **Source Selection**: Built around the gold-standard MIT-BIH Arrhythmia Database.
- **Conversion Phase**: Raw data (`.dat` and `.hea` files) is converted into interoperable formats like EDF (European Data Format).
- **Signal Filtering**: Applies a **0.5Hz - 40Hz Butterworth Bandpass Filter** to mitigate baseline wander and limit powerline interference noise.
- **Normalization**: Z-Score normalization per-segment allows robust morphological feature extraction over amplitude thresholds.

### 2. Dual-Mode Deep Learning Architecture 
The application guarantees mathematical isolation natively handling differing geometries using independent processing environments:

#### A. Raw Sequence Engine (DSNN)
- Realized as a Deep Convolutional SNN specifically structured for sequential temporal parsing.
- Employs **Peak-Triggered Segmentation**, centering focus cleanly along the R-Peak utilizing `.qrs` annotations.
- Addresses the rareness of life-threatening arrhythmias via statistically balanced subset iterations and focal sequence loss.

#### B. Tabular Morphology Pipeline (CSV-MLP)
- Extracts morphological attributes scaled accurately via `RobustScaler` (protecting tabular data from intense outlier variance).
- Features an ultra-shallow Residual Block structure (`Input -> 64 -> 32 -> 16 -> num_classes`) designed natively to halt tabular point-memorization (overfitting).
- Mitigates class dominance (where Normal overrides Ventricular classes) by wrapping computed `balanced` class weights with an inverse square root cap restrictor.
- Natively forces healthy diagnostic boundaries with strict `0.1` label smoothing, `AdamW` L2 structural decays, and `CosineAnnealingLR` schedules ensuring gradient traversals peak flawlessly around ~85%.

### 3. Inference & Diagnostics
- Outputs predictions sorted among 5 main classifications across distinct modalities:
  1. Normal Sinus Rhythm (N)
  2. Supraventricular Arrhythmia (S)
  3. Ventricular Arrhythmia (V)
  4. Fusion Beat (F)
  5. Unknown / Unclassifiable (Q)
- Delivers a primary diagnosis with an associated **confidence/probability score**. Detailed segments map specific regions contributing to the classification.

---

## 📂 Folder Structure

```text
Arrhythmia-Detection/
├── backend/                   # Python / Flask REST API & ML Services
│   ├── app.py                 # Core app entry point
│   ├── routes/                # Endpoints (api.py, web.py)
│   ├── services/              # Core logic & ML Pipelines (train_dsnn.py, converter.py, ecg_reader.py)
│   ├── models/                # Checkpoints & serialized PyTorch models (.pth)
│   ├── results/               # Persisted Analysis Reports (results.json, training_results.json)
│   ├── uploads/               # Temporary parsing directory for incoming ECG readings
│   └── images/                # Cached & programmatic generated Training Visualizations (training_history.png, confusion_matrix.png, ecg_daigrams)
├── frontend/                  # React + Vite Frontend UI
│   ├── src/
│   │   ├── pages/             # Main Views (Dashboard, History, Home, ModelTraining, Upload, Results, Settings)
│   │   ├── components/        # Isolated UI pieces (Sidebar, Header, LoadingSpinner)
│   │   ├── services/          # API Services (api.js)
│   │   ├── App.jsx            # Routing Rules & Theme Provider Context
│   │   ├── App.css            # Scoped layout styling and transitions
|   |   ├── main.jsx           # Entry point
|   |   ├── index.css          # Global styles
|   |   └── index.html         # HTML template
|   ├── assets/            # Static assets (images, fonts, etc.)
|   ├── vite.config.js     # Build tooling config
└── Dataset/                   # Persistent ECG Data Repository
    ├── MIT-BIH/               # Benchmark training records (.edf, .qrs)
    └── test/                  # Test set mappings for independent inference
```

---

## 🛠️ Tech Stack & Prerequisites

### Application Layers
- **Frontend**: React.js, Vite, React-Router-DOM, Recharts (Visualizations), Axios (Data fetching), Lucide React (Icons).
- **Backend**: Flask (API Routing), PyTorch (SNN Definition/Training), WFDB (Bio-signal Parsing), SciPy/NumPy.

### Prerequisites needed heavily exclusively locally:
- **Python** (version 3.10+)
- **Node.js** (version 18+)
- **NPM / Yarn**
- **CUDA Toolkit** (Optional but recommended for rendering the DSNN iterations locally much quicker).

---

## ⚙️ Setup & Installations

Execute the following actions chronologically to boot up the environment locally.

### 1. Initialize the Backend
Run the enclosed services spanning the data pipelines and active API.

```bash
# Navigate to the backend directory
cd backend

# (Optional but recommended) Create and activate a Virtual Environment
python -m venv venv

# Install the Python dependencies
pip install -r requirements.txt
```

### 2. Initialize the Frontend
Install user-interface dependencies necessary to spin the React development server.

```bash
# Navigate to the frontend directory
cd frontend

# Install exact UI dependencies
npm install
```

---

## 🎮 Commands: Running the Application

To ensure smooth runtime performance, run the core servers across two concurrent active terminals:

**Terminal 1 (Backend - API & Model Router)**
```bash
cd backend
# Activate the virtual environment
# On Windows
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate

python app.py

```
> The API layer initializes on `http://localhost:5000`.

**Terminal 2 (Frontend - CLI & UI Wrapper)**
```bash
cd frontend
npm run dev
```
> The Web Application will normally initialize and compile on `http://localhost:5173`.
