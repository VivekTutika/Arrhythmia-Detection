# Arrhythmia Detection Web Application

Early Detection of Arrhythmia using Deep Spiking Neural Network (DSNN) - Web Application

## Project Structure

```
Arrhythmia-Detection/
├── frontend/                 # Vite + React frontend
│   ├── src/
│   │   ├── components/       # Reusable UI components
│   │   ├── pages/           # Page components
│   │   ├── App.jsx          # Main app component
│   │   ├── App.css          # Global styles
│   │   └── index.css        # Base styles
│   ├── package.json
│   └── vite.config.js
├── backend/                  # Flask backend API
│   ├── routes/
│   │   ├── api.py           # API endpoints
│   │   └── web.py           # Web routes
│   ├── app.py               # Flask app entry point
│   └── requirements.txt
├── Dataset/                  # ECG dataset
│   ├── edf/                 # EDF files
│   └── qrs/                 # QRS annotations
└── backend/                  # Original DSNN model code
    ├── dsnn_example.py
    ├── train_dsnn.py
    └── ...
```

## Features

- **Dashboard**: Overview of arrhythmia detection statistics with charts
- **Upload**: Upload EDF ECG files for analysis
- **Results**: Detailed analysis results with confidence scores
- **History**: View all previous analysis results
- **Settings**: Configure model parameters and preferences
- **Dark/Light Mode**: Toggle between themes

## Prerequisites

- Node.js 18+ 
- Python 3.8+
- pip

## Installation

### Backend (Flask API)

```bash
cd backend
pip install -r requirements.txt
```

### Frontend (React)

```bash
cd frontend
npm install
```

## Running the Application

### Option 1: Run Both Separately

**Terminal 1 - Backend:**
```bash
cd backend
python app.py
```
API will run at http://localhost:5000

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```
Frontend will run at http://localhost:3000

### Option 2: Run with Production Build

1. Build frontend:
```bash
cd frontend
npm run build
```

2. Run backend:
```bash
cd backend
python app.py
```

Access at http://localhost:5000

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/dashboard` | Get dashboard statistics |
| POST | `/api/analyze` | Analyze ECG file |
| GET | `/api/results` | Get all results (paginated) |
| GET | `/api/results/:id` | Get specific result |
| DELETE | `/api/results/:id` | Delete a result |
| GET | `/health` | Health check |

## DSNN Model Integration

The web app currently uses mock predictions. To integrate your trained DSNN model:

1. Place your trained model in `backend/models/`
2. Update `process_ecg_file()` in `backend/routes/api.py` to:
   - Load your model
   - Process EDF files
   - Run inference
   - Return predictions

## Classes

The model detects 6 classes:
1. Normal Sinus Rhythm
2. Atrial Fibrillation
3. Ventricular Arrhythmia
4. Conduction Block
5. Premature Contraction
6. ST Segment Abnormality

## Tech Stack

### Frontend
- React 18
- Vite
- React Router DOM
- Recharts
- Lucide React Icons
- Axios

### Backend
- Flask
- Flask-CORS
- NumPy
- PyEDFlib (for EDF files)
- WFDB (for QRS annotations)

### Original DSNN (Backend/Machine Learning)
- PyTorch
- Custom CNN architecture
- Data augmentation via sequence shifting
- Voting mechanism for predictions

## License

MIT License

