import axios from 'axios';

// In development, use the proxy (no CORS issues)
// In production, use the environment variable
const isDevelopment = import.meta.env.DEV;
const API_BASE_URL = isDevelopment ? '' : (import.meta.env.VITE_API_URL || 'http://localhost:5000');

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 second timeout for ECG processing
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add any auth tokens if needed
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// ============ API Methods ============

// Analyze ECG file
export const analyzeECG = async (file, patientInfo = {}) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('patient_info', JSON.stringify(patientInfo));
  
  const response = await apiClient.post('/api/analyze', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

// Get analysis result by ID
export const getResult = async (id) => {
  const response = await apiClient.get(`/api/results/${id}`);
  return response.data;
};

// Get all results
export const getAllResults = async (page = 1, status = '') => {
  const response = await apiClient.get('/api/results', {
    params: { page, status },
  });
  return response.data;
};

// Get dashboard stats
export const getDashboardStats = async () => {
  const response = await apiClient.get('/api/dashboard');
  return response.data;
};

// Delete a result
export const deleteResult = async (id) => {
  const response = await apiClient.delete(`/api/results/${id}`);
  return response.data;
};

// Delete all results
export const deleteAllResults = async () => {
  const response = await apiClient.delete('/api/results');
  return response.data;
};

// Convert MIT-BIH dataset
export const convertMitbih = async () => {
  const response = await apiClient.post('/api/convert-mitbih');
  return response.data;
};

// Train model
export const trainModel = async (datasetPath, epochs) => {
  const response = await apiClient.post('/api/train-model', {
    dataset_path: datasetPath,
    epochs: epochs
  });
  return response.data;
};

// Get training status
export const getTrainingStatus = async () => {
  const response = await apiClient.get('/api/training-status');
  return response.data;
};

// Stop training
export const stopTraining = async () => {
  const response = await apiClient.post('/api/stop-training');
  return response.data;
};

export default apiClient;
