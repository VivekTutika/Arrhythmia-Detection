// API Configuration
// Uses environment variable VITE_API_URL, defaults to localhost for development
// In development, use proxy (empty string), in production use full URL

const isDevelopment = import.meta.env.DEV;
const API_BASE_URL = isDevelopment ? '' : (import.meta.env.VITE_API_URL || 'http://localhost:5000');

export const API_ENDPOINTS = {
  ANALYZE: `/api/analyze`,
  RESULTS: (id) => `/api/results/${id}`,
  ALL_RESULTS: `/api/results`,
  DASHBOARD: `/api/dashboard`,
  UPLOAD: `/api/analyze`,
};

export default API_BASE_URL;
