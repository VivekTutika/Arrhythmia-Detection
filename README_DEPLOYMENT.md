# Arrhythmia Detection - Deployment Guide

## Overview
This project is a web-based arrhythmia detection system using Deep Spiking Neural Network (DSNN). It consists of:
- **Backend**: Flask API (deploy to Heroku/Render)
- **Frontend**: React + Vite (deploy to Vercel)

## Deployment Platforms

### Backend Deployment (Heroku/Render)

1. **Prepare the backend**:
   
```
bash
   cd backend
   pip install -r requirements.txt
   
```

2. **Deploy to Heroku**:
   
```
bash
   # Install Heroku CLI
   heroku login
   heroku create your-app-name
   git add .
   git commit -m "Deploy backend"
   git push heroku main
   
```

3. **Deploy to Render**:
   - Connect your GitHub repository
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `python app.py`
   - Add environment variable: `PORT=5000`

### Frontend Deployment (Vercel)

1. **Configure environment variables**:
   - Copy `.env.example` to `.env`
   - Update `VITE_API_URL` to your backend URL:
     
```
     VITE_API_URL=https://your-backend-app.onrender.com
     
```

2. **Deploy to Vercel**:
   
```
bash
   cd frontend
   npm install
   npm run build
   vercel deploy --prod
   
```

   Or connect your GitHub repository to Vercel for automatic deployments.

## Environment Variables

### Backend (.env)
```
PORT=5000
FLASK_DEBUG=False
SECRET_KEY=your-secret-key
```

### Frontend (.env)
```
VITE_API_URL=http://localhost:5000  # Development
VITE_API_URL=https://your-backend.onrender.com  # Production
```

## Project Structure
```
Arrhythmia-Detection/
├── backend/
│   ├── app.py              # Flask application
│   ├── routes/
│   │   ├── api.py         # API routes
│   │   └── web.py        # Web routes
│   ├── requirements.txt   # Python dependencies
│   ├── Procfile          # Heroku deployment
│   └── runtime.txt       # Python version
├── frontend/
│   ├── src/
│   │   ├── pages/        # React pages
│   │   ├── services/
│   │   │   └── api.js   # API service
│   │   └── config.js    # API configuration
│   ├── vercel.json       # Vercel config
│   └── .env              # Environment variables
└── Dataset/              # ECG data files
```

## Running Locally

### Backend
```
bash
cd backend
python app.py
# API runs at http://localhost:5000
```

### Frontend
```
bash
cd frontend
npm install
npm run dev
# Frontend runs at http://localhost:5173
```

## API Endpoints
- `GET /api/dashboard` - Dashboard statistics
- `POST /api/analyze` - Upload and analyze ECG
- `GET /api/results` - List all results
- `GET /api/results/<id>` - Get specific result
- `DELETE /api/results/<id>` - Delete result
- `DELETE /api/results` - Delete all results
- `GET /health` - Health check
