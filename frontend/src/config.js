// API Configuration
// Uses environment variable VITE_API_URL, defaults to localhost for development

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

export const API_ENDPOINTS = {
  ANALYZE: `${API_BASE_URL}/api/analyze`,
  RESULTS: (id) => `${API_BASE_URL}/api/results/${id}`,
  ALL_RESULTS: `${API_BASE_URL}/api/results`,
  DASHBOARD: `${API_BASE_URL}/api/dashboard`,
  UPLOAD: `${API_BASE_URL}/api/analyze`,
};

export default API_BASE_URL;
