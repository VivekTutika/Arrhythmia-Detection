import { useState } from 'react';
import { Save, RefreshCw, Info, HardDrive, Brain, Bell, Shield, CheckCircle, Edit2, X } from 'lucide-react';

const defaultSettings = {
  modelPath: './models/dsnn_model.pth',
  dataPath: './Dataset/MIT-BIH',
  batchSize: 32,
  numWorkers: 4,
  confidenceThreshold: 60,
  autoSave: true,
  notifications: true,
  darkMode: false
};

const Settings = () => {
  // Initialize state from localStorage or use defaults
  const [settings, setSettings] = useState(() => {
    const saved = localStorage.getItem('appSettings');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        console.error('Failed to parse settings', e);
      }
    }
    return defaultSettings;
  });

  const [modalMessage, setModalMessage] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [isEditing, setIsEditing] = useState(false);

  const handleSave = () => {
    localStorage.setItem('appSettings', JSON.stringify(settings));
    setModalMessage('Settings saved successfully!');
    setShowModal(true);
    setIsEditing(false);
  };

  const handleReset = () => {
    setSettings(defaultSettings);
    localStorage.setItem('appSettings', JSON.stringify(defaultSettings));
    setModalMessage('Settings reset to defaults');
    setShowModal(true);
    setIsEditing(false);
  };
  
  const handleCancel = () => {
    // Revert settings back to the saved state in localStorage
    const saved = localStorage.getItem('appSettings');
    if (saved) {
      try {
        setSettings(JSON.parse(saved));
      } catch (e) {
        setSettings(defaultSettings);
      }
    } else {
      setSettings(defaultSettings);
    }
    setIsEditing(false);
  };

  return (
    <div className="settings-page">
      {/* Success Modal */}
      {showModal && (
        <div className="modal-overlay">
          <div className="modal-content" style={{ maxWidth: '400px', textAlign: 'center' }}>
            <div style={{ marginBottom: '16px' }}>
              <div style={{ 
                width: '48px', height: '48px', borderRadius: '50%', 
                background: 'rgba(16, 185, 129, 0.1)', color: '#10b981', 
                display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto' 
              }}>
                <CheckCircle size={28} />
              </div>
            </div>
            <h3 style={{ marginTop: 0, marginBottom: '8px' }}>Success</h3>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '24px' }}>
              {modalMessage}
            </p>
            <button 
              className="btn btn-primary"
              onClick={() => setShowModal(false)}
              style={{ width: '100%' }}
            >
              Close
            </button>
          </div>
        </div>
      )}

      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1>Settings</h1>
          <p className="text-secondary">Configure application preferences</p>
        </div>
        {!isEditing && (
          <button className="btn btn-primary" onClick={() => setIsEditing(true)}>
            <Edit2 size={18} />
            Edit Settings
          </button>
        )}
      </div>

      <div style={{ display: 'grid', gap: '24px', maxWidth: '800px' }}>
        {/* Model Settings */}
        <div className="card">
          <div className="card-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <Brain size={24} style={{ color: 'var(--primary-color)' }} />
              <div>
                <h3 className="card-title">Model Configuration                <p className="card-subtitle">Configure DSNN model parameters</p>
             </h3>
 </div>
            </div>
          </div>
          
          <div style={{ display: 'grid', gap: '16px' }}>
            <div className="form-group">
              <label className="form-label">Model File Path</label>
              <input
                type="text"
                className="form-input"
                value={settings.modelPath}
                onChange={(e) => setSettings({...settings, modelPath: e.target.value})}
                disabled={!isEditing}
                style={{ opacity: !isEditing ? 0.7 : 1, cursor: !isEditing ? 'not-allowed' : 'text' }}
              />
            </div>
            
            <div className="form-group">
              <label className="form-label">Batch Size</label>
              <input
                type="number"
                className="form-input"
                value={settings.batchSize}
                onChange={(e) => setSettings({...settings, batchSize: parseInt(e.target.value)})}
                disabled={!isEditing}
                style={{ opacity: !isEditing ? 0.7 : 1, cursor: !isEditing ? 'not-allowed' : 'text' }}
              />
            </div>
            
            <div className="form-group">
              <label className="form-label">Number of Workers</label>
              <input
                type="number"
                className="form-input"
                value={settings.numWorkers}
                onChange={(e) => setSettings({...settings, numWorkers: parseInt(e.target.value)})}
                disabled={!isEditing}
                style={{ opacity: !isEditing ? 0.7 : 1, cursor: !isEditing ? 'not-allowed' : 'text' }}
              />
            </div>
          </div>
        </div>

        {/* Data Settings */}
        <div className="card">
          <div className="card-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <HardDrive size={24} style={{ color: 'var(--primary-color)' }} />
              <div>
                <h3 className="card-title">Data Settings</h3>
                <p className="card-subtitle">Configure data paths and storage</p>
              </div>
            </div>
          </div>
          
          <div style={{ display: 'grid', gap: '16px' }}>
            <div className="form-group">
              <label className="form-label">Default Data Path</label>
              <input
                type="text"
                className="form-input"
                value={settings.dataPath}
                onChange={(e) => setSettings({...settings, dataPath: e.target.value})}
                disabled={!isEditing}
                style={{ opacity: !isEditing ? 0.7 : 1, cursor: !isEditing ? 'not-allowed' : 'text' }}
              />
            </div>
            
            <div className="form-group">
              <label className="form-label">Confidence Threshold (%)</label>
              <input
                type="number"
                className="form-input"
                min="0"
                max="100"
                value={settings.confidenceThreshold}
                onChange={(e) => setSettings({...settings, confidenceThreshold: parseInt(e.target.value)})}
                disabled={!isEditing}
                style={{ opacity: !isEditing ? 0.7 : 1, cursor: !isEditing ? 'not-allowed' : 'text' }}
              />
              <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Minimum confidence required to display a prediction
              </p>
            </div>
          </div>
        </div>

        {/* App Settings */}
        <div className="card">
          <div className="card-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <Shield size={24} style={{ color: 'var(--primary-color)' }} />
              <div>
                <h3 className="card-title">Application Settings</h3>
                <p className="card-subtitle">General application preferences</p>
              </div>
            </div>
          </div>
          
          <div style={{ display: 'grid', gap: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px', background: 'var(--background-secondary)', borderRadius: '8px' }}>
              <div>
                <div style={{ fontWeight: '500' }}>Auto-save Results</div>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Automatically save analysis results</div>
              </div>
              <label style={{ position: 'relative', display: 'inline-block', width: '48px', height: '24px' }}>
                <input
                  type="checkbox"
                  checked={settings.autoSave}
                  onChange={(e) => setSettings({...settings, autoSave: e.target.checked})}
                  disabled={!isEditing}
                  style={{ opacity: 0, width: 0, height: 0 }}
                />
                <span style={{
                  position: 'absolute',
                  cursor: isEditing ? 'pointer' : 'not-allowed',
                  top: 0, left: 0, right: 0, bottom: 0,
                  backgroundColor: settings.autoSave ? 'var(--primary-color)' : 'var(--border-color)',
                  borderRadius: '24px',
                  transition: '0.3s',
                  opacity: !isEditing ? 0.6 : 1,
                  pointerEvents: isEditing ? 'auto' : 'none'
                }}>
                  <span style={{
                    position: 'absolute',
                    content: '""',
                    height: '18px',
                    width: '18px',
                    left: settings.autoSave ? '27px' : '3px',
                    bottom: '3px',
                    backgroundColor: 'white',
                    borderRadius: '50%',
                    transition: '0.3s'
                  }} />
                </span>
              </label>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px', background: 'var(--background-secondary)', borderRadius: '8px' }}>
              <div>
                <div style={{ fontWeight: '500' }}>Notifications</div>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Enable push notifications</div>
              </div>
              <label style={{ position: 'relative', display: 'inline-block', width: '48px', height: '24px' }}>
                <input
                  type="checkbox"
                  checked={settings.notifications}
                  onChange={(e) => setSettings({...settings, notifications: e.target.checked})}
                  disabled={!isEditing}
                  style={{ opacity: 0, width: 0, height: 0 }}
                />
                <span style={{
                  position: 'absolute',
                  cursor: isEditing ? 'pointer' : 'not-allowed',
                  top: 0, left: 0, right: 0, bottom: 0,
                  backgroundColor: settings.notifications ? 'var(--primary-color)' : 'var(--border-color)',
                  borderRadius: '24px',
                  transition: '0.3s',
                  opacity: !isEditing ? 0.6 : 1,
                  pointerEvents: isEditing ? 'auto' : 'none'
                }}>
                  <span style={{
                    position: 'absolute',
                    content: '""',
                    height: '18px',
                    width: '18px',
                    left: settings.notifications ? '27px' : '3px',
                    bottom: '3px',
                    backgroundColor: 'white',
                    borderRadius: '50%',
                    transition: '0.3s'
                  }} />
                </span>
              </label>
            </div>
          </div>
        </div>

        {/* About */}
        <div className="card">
          <div className="card-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <Info size={24} style={{ color: 'var(--primary-color)' }} />
              <div>
                <h3 className="card-title">About</h3>
                <p className="card-subtitle">Application information</p>
              </div>
            </div>
          </div>
          
          <div style={{ display: 'grid', gap: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Version</span>
              <span>1.0.0</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Model</span>
              <span>DSNN (Deep Spiking Neural Network)</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Classes</span>
              <span>6 (Normal + 5 Arrhythmia Types)</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Framework</span>
              <span>PyTorch + React</span>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        {isEditing && (
          <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
            <button className="btn btn-secondary" onClick={handleReset}>
              <RefreshCw size={18} />
              Reset to Defaults
            </button>
            <button className="btn btn-secondary" onClick={handleCancel}>
              <X size={18} />
              Cancel
            </button>
            <button className="btn btn-primary" onClick={handleSave}>
              <Save size={18} />
              Save Settings
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Settings;

