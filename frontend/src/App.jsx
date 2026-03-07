import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Dashboard from './pages/Dashboard';
import Upload from './pages/Upload';
import Results from './pages/Results';
import History from './pages/History';
import Settings from './pages/Settings';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import LoadingSpinner from './components/LoadingSpinner';
import './App.css';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [theme, setTheme] = useState('light');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);
  const toggleTheme = () => setTheme(theme === 'light' ? 'dark' : 'light');

  return (
    <Router>
      <div className="app-container">
        <Sidebar isOpen={sidebarOpen} />
        <div className={`main-content ${sidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
          <Header 
            toggleSidebar={toggleSidebar} 
            toggleTheme={toggleTheme}
            theme={theme}
          />
          <div className="content-area">
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route 
                path="/dashboard" 
                element={<Dashboard isLoading={isLoading} setIsLoading={setIsLoading} />} 
              />
              <Route 
                path="/upload" 
                element={<Upload isLoading={isLoading} setIsLoading={setIsLoading} />} 
              />
              <Route 
                path="/results/:id" 
                element={<Results isLoading={isLoading} setIsLoading={setIsLoading} />} 
              />
              <Route 
                path="/history" 
                element={<History isLoading={isLoading} setIsLoading={setIsLoading} />} 
              />
              <Route 
                path="/settings" 
                element={<Settings isLoading={isLoading} setIsLoading={setIsLoading} />} 
              />
            </Routes>
          </div>
        </div>
        {isLoading && <LoadingSpinner />}
        <Toaster 
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: theme === 'dark' ? '#333' : '#fff',
              color: theme === 'dark' ? '#fff' : '#333',
            },
          }}
        />
      </div>
    </Router>
  );
}

export default App;

