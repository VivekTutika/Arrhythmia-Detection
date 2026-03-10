import { useState, useEffect, useRef } from 'react';
import { 
  Activity, 
  FileText, 
  Play, 
  Square,
  RefreshCw,
  CheckCircle,
  XCircle,
  Maximize2,
  Minimize2
} from 'lucide-react';
import { convertMitbih, trainModel, getTrainingStatus, stopTraining } from '../services/api';

const ModelTraining = ({ setIsLoading }) => {
  const [activeTab, setActiveTab] = useState('training');
  const [converting, setConverting] = useState(false);
  const [conversionResult, setConversionResult] = useState(null);
  const [training, setTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [datasetPath, setDatasetPath] = useState('Dataset/MIT-BIH');
  const [epochs, setEpochs] = useState(50);
  const [expandedImage, setExpandedImage] = useState(null);
  const [showStartConfirm, setShowStartConfirm] = useState(false);
  const [showStopConfirm, setShowStopConfirm] = useState(false);
  const pollIntervalRef = useRef(null);

  useEffect(() => {
    // Reset isLoading on mount to prevent stuck spinner
    setIsLoading(false);
    
    fetchTrainingStatus();
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  const fetchTrainingStatus = async () => {
    try {
      const status = await getTrainingStatus();
      setTrainingStatus(status);
      
      // Update local training state based on status
      if (status.status === 'running') {
        setTraining(true);
        // Keep isLoading true if training is running
      } else if (status.status === 'completed') {
        setTraining(false);
        setIsLoading(false);
      } else if (status.status === 'stopped' || status.status === 'failed') {
        setTraining(false);
        setIsLoading(false);
      } else {
        // not_started or any other state
        setTraining(false);
        setIsLoading(false);
      }
    } catch (error) {
      console.error('Error fetching training status:', error);
      setIsLoading(false);
    }
  };

  const handleConvert = async () => {
    setConverting(true);
    setConversionResult(null);
    setIsLoading(true);
    
    try {
      const result = await convertMitbih();
      setConversionResult(result);
    } catch (error) {
      setConversionResult({
        success: false,
        error: error.response?.data?.error || error.message
      });
    } finally {
      setConverting(false);
      setIsLoading(false);
    }
  };

  const startPolling = () => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }
    
    pollIntervalRef.current = setInterval(async () => {
      const status = await getTrainingStatus();
      setTrainingStatus(status);
      
      if (status.status === 'completed') {
        setTraining(false);
        setIsLoading(false);
        clearInterval(pollIntervalRef.current);
      } else if (status.status === 'stopped') {
        setTraining(false);
        setIsLoading(false);
        clearInterval(pollIntervalRef.current);
      } else if (status.status === 'failed') {
        setTraining(false);
        setIsLoading(false);
        clearInterval(pollIntervalRef.current);
      }
    }, 2000);
  };

  const handleStartTraining = async () => {
    setShowStartConfirm(false);
    
    // Clear previous training status and set training to true immediately
    setTrainingStatus(null);
    setTraining(true);
    setIsLoading(true);
    
    try {
      const result = await trainModel(datasetPath, epochs);
      console.log('Training started:', result);
      
      // Start polling for status updates
      startPolling();
      
    } catch (error) {
      console.error('Training error:', error);
      setTraining(false);
      setIsLoading(false);
    }
  };

  const handleStopTraining = async () => {
    setShowStopConfirm(false);
    
    try {
      const result = await stopTraining();
      console.log('Stop training result:', result);
      // The polling will handle updating the status
    } catch (error) {
      console.error('Error stopping training:', error);
    }
  };

  const handleTrainClick = () => {
    if (training) {
      setShowStopConfirm(true);
    } else {
      setShowStartConfirm(true);
    }
  };

  return (
    <div className="model-training-page">
      {/* Tab Navigation - Enhanced Switchable Tabs */}
      <div className="tabs-container" style={{ marginBottom: '4px' }}>
        <div className="enhanced-tabs">
          <button 
            className={`enhanced-tab ${activeTab === 'preprocessing' ? 'active' : ''}`}
            onClick={() => setActiveTab('preprocessing')}
          >
            <FileText size={18} />
            <span>Pre-Processing</span>
            <div className="tab-indicator"></div>
          </button>
          <button 
            className={`enhanced-tab ${activeTab === 'training' ? 'active' : ''}`}
            onClick={() => setActiveTab('training')}
          >
            <Activity size={18} />
            <span>Model Training</span>
            <div className="tab-indicator"></div>
          </button>
        </div>
      </div>

      {/* Pre-Processing Tab */}
      {activeTab === 'preprocessing' && (
        <div className="card">
          <div className="card-header">
            <div>
              <h3 className="card-title">MIT-BIH Dataset Conversion</h3>
              <p className="card-subtitle">Convert MIT-BIH dataset files (.hea, .dat, .atr) to EDF and QRS format</p>
            </div>
          </div>
          
          <div>
            <div style={{ 
              display: 'flex', 
              flexDirection: 'column', 
              alignItems: 'center', 
              gap: '16px',
              padding: '32px',
              background: 'var(--background-secondary)',
              borderRadius: '12px'
            }}>
              <div style={{ 
                width: '64px', 
                height: '64px', 
                borderRadius: '50%', 
                background: 'var(--primary-color)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <RefreshCw size={32} color="white" />
              </div>
              
              {/* Processing Time Info */}
              <div className="processing-info">
                <div className="processing-info-icon">
                  <RefreshCw size={20} color="white" />
                </div>
                <div className="processing-info-content">
                  <div className="processing-info-title">Estimated Processing Time</div>
                  <div className="processing-info-text">
                    Conversion typically takes 2-5 minutes depending on the number of files. 
                    The process runs in the background and you'll be notified upon completion.
                  </div>
                </div>
              </div>
              
              <button 
                className="btn btn-primary"
                onClick={handleConvert}
                disabled={converting}
                style={{ minWidth: '200px' }}
              >
                {converting ? (
                  <>
                    <RefreshCw size={18} className="spin" />
                    Converting...
                  </>
                ) : (
                  <>
                    <Play size={18} />
                    Convert MIT-BIH Data
                  </>
                )}
              </button>

              {/* Progress Indicator during conversion */}
              {converting && (
                <div className="processing-progress">
                  <div className="processing-progress-bar">
                    <div className="processing-progress-fill" style={{ width: '60%' }}></div>
                  </div>
                  <div className="processing-progress-text">Processing files...</div>
                </div>
              )}
            </div>

            {/* Conversion Result */}
            {conversionResult && (
              <div style={{ marginTop: '24px' }}>
                {conversionResult.success ? (
                  <div style={{ 
                    padding: '16px', 
                    background: 'rgba(16, 185, 129, 0.1)', 
                    border: '1px solid #10b981',
                    borderRadius: '8px',
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '12px'
                  }}>
                    <CheckCircle size={24} style={{ color: '#10b981' }} />
                    <div>
                      <h4 style={{ margin: '0 0 8px 0', color: '#10b981' }}>Conversion Successful!</h4>
                      <p style={{ margin: 0, color: 'var(--text-secondary)' }}>
                        {conversionResult.message}
                      </p>
                      {conversionResult.results && (
                        <div style={{ marginTop: '12px', display: 'flex', gap: '24px' }}>
                          <div>
                            <span style={{ fontWeight: 'bold' }}>EDF Files:</span> {conversionResult.results.edf.success} created
                          </div>
                          <div>
                            <span style={{ fontWeight: 'bold' }}>QRS Files:</span> {conversionResult.results.qrs.success} created
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div style={{ 
                    padding: '16px', 
                    background: 'rgba(239, 68, 68, 0.1)', 
                    border: '1px solid #ef4444',
                    borderRadius: '8px',
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '12px'
                  }}>
                    <XCircle size={24} style={{ color: '#ef4444' }} />
                    <div>
                      <h4 style={{ margin: '0 0 8px 0', color: '#ef4444' }}>Conversion Failed</h4>
                      <p style={{ margin: 0, color: 'var(--text-secondary)' }}>
                        {conversionResult.error || 'An error occurred during conversion'}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Model Training Tab */}
      {activeTab === 'training' && (
        <div className="card">
          <div className="card-header">
            <div>
              <h3 className="card-title">Train DSNN Model</h3>
              <p className="card-subtitle">Configure and train the Deep Spiking Neural Network for arrhythmia detection</p>
            </div>
          </div>
          
          <div style={{ padding: '8px' }}>
            {/* Training Configuration */}
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', 
              gap: '24px',
              marginBottom: '24px'
            }}>
              <div className="form-group">
                <label className="form-label">Dataset Path</label>
                <input
                  type="text"
                  className="form-input"
                  value={datasetPath}
                  onChange={(e) => setDatasetPath(e.target.value)}
                  placeholder="Dataset/edf"
                  disabled={training}
                />
                <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                  Path to the folder containing EDF files
                </p>
              </div>
              
              <div className="form-group">
                <label className="form-label">Number of Epochs</label>
                <input
                  type="number"
                  className="form-input"
                  value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value) || 50)}
                  min={1}
                  max={5000}
                  disabled={training}
                />
                <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                  Training iterations (recommended: 50-200) (Min: 1, Max: 5000)
                </p>
              </div>
            </div>

            {/* Processing Time Info for Training */}
            <div className="processing-info">
              <div className="processing-info-icon">
                <Activity size={20} color="white" />
              </div>
              <div className="processing-info-content">
                <div className="processing-info-title">Estimated Processing Time</div>
                <div className="processing-info-text">
                  Training with {epochs} epochs typically takes {Math.ceil(epochs / 10)}-{Math.ceil(epochs / 5)} minutes. 
                  Time varies based on dataset size and your device performance.
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '24px' }}>
              <button 
                className={`btn ${training ? 'btn-danger' : 'btn-primary'}`}
                onClick={handleTrainClick}
                disabled={!datasetPath && !training}
                style={{ minWidth: '200px' }}
              >
                {training ? (
                  <>
                    <Square size={18} />
                    Stop Training
                  </>
                ) : (
                  <>
                    <Play size={18} />
                    Start Training
                  </>
                )}
              </button>
            </div>

            {/* Progress Indicator during training */}
            {training && (
              <div className="processing-progress">
                <div className="processing-progress-bar">
                  <div className="processing-progress-fill" style={{ width: '45%' }}></div>
                </div>
                <div className="processing-progress-text">Training model... (Click "Stop Training" to cancel)</div>
              </div>
            )}

            {/* Training Results - Only show when training is completed/stopped/failed */}
            {trainingStatus && !training && trainingStatus.status !== 'not_started' && (
              <div>
                <h4 style={{ marginBottom: '16px' }}>Training Results</h4>
                
                {/* Show completed message */}
                {trainingStatus.status === 'completed' && (
                  <div style={{ 
                    padding: '16px', 
                    background: 'rgba(16, 185, 129, 0.1)', 
                    border: '1px solid #10b981',
                    borderRadius: '8px',
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '12px',
                    marginBottom: '16px'
                  }}>
                    <CheckCircle size={24} style={{ color: '#10b981' }} />
                    <div>
                      <h4 style={{ margin: '0 0 8px 0', color: '#10b981' }}>Training Completed Successfully!</h4>
                      <p style={{ margin: 0, color: 'var(--text-secondary)' }}>
                        The model has been trained and is ready for use.
                      </p>
                    </div>
                  </div>
                )}
                
                {/* Show stopped/failed message */}
                {trainingStatus.status === 'stopped' && (
                  <div style={{
                    padding: '16px', 
                    background: 'rgba(251, 191, 36, 0.1)', 
                    border: '1px solid #fbbf24',
                    borderRadius: '8px',
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '12px',
                    marginBottom: '16px'
                  }}>
                    <XCircle size={24} style={{ color: '#fbbf24' }} />
                    <div>
                      <h4 style={{ margin: '0 0 8px 0', color: '#fbbf24' }}>Training Stopped</h4>
                      <p style={{ margin: 0, color: 'var(--text-secondary)' }}>
                        The training was stopped by the user. No model or visualization files were generated.
                      </p>
                    </div>
                  </div>
                )}

                {trainingStatus.status === 'failed' && trainingStatus.error && (
                  <div style={{ 
                    padding: '16px', 
                    background: 'rgba(239, 68, 68, 0.1)', 
                    border: '1px solid #ef4444',
                    borderRadius: '8px',
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '12px',
                    marginBottom: '16px'
                  }}>
                    <XCircle size={24} style={{ color: '#ef4444' }} />
                    <div>
                      <h4 style={{ margin: '0 0 8px 0', color: '#ef4444' }}>Training Failed</h4>
                      <p style={{ margin: 0, color: 'var(--text-secondary)' }}>
                        {trainingStatus.error}
                      </p>
                    </div>
                  </div>
                )}
                
                {trainingStatus.image_files && trainingStatus.image_files.length > 0 ? (
                  <div style={{ 
                    display: 'grid', 
                    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', 
                    gap: '16px' 
                  }}>
                    {trainingStatus.image_files.map((img, idx) => (
                      <div 
                        key={idx}
                        style={{ 
                          background: 'var(--background-secondary)', 
                          borderRadius: '8px',
                          overflow: 'hidden'
                        }}
                      >
                        <div style={{ padding: '12px', borderBottom: '1px solid var(--border-color)' }}>
                          <span style={{ fontWeight: '500' }}>
                            {img.includes('training') ? 'Training History' : 'Confusion Matrix'}
                          </span>
                        </div>
                        <div style={{ position: 'relative' }}>
                          <img 
                            src={img} 
                            alt={img} 
                            style={{ width: '100%', height: 'auto', display: 'block' }}
                          />
                          <button
                            className="btn btn-sm btn-secondary"
                            onClick={() => setExpandedImage(img)}
                            style={{ 
                              position: 'absolute', 
                              top: '8px', 
                              right: '8px',
                              background: 'rgba(0,0,0,0.5)',
                              color: 'white'
                            }}
                          >
                            <Maximize2 size={14} />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ 
                    padding: '24px', 
                    textAlign: 'center',
                    background: 'var(--background-secondary)',
                    borderRadius: '8px'
                  }}>
                    <p style={{ color: 'var(--text-secondary)' }}>
                      {trainingStatus.status === 'completed' 
                        ? 'Training completed. No visualization files generated yet.'
                        : 'No training results available. Start training to see results.'}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Start Training Confirmation Modal */}
      {showStartConfirm && (
        <div className="modal-overlay">
          <div className="modal-content" style={{ maxWidth: '400px' }}>
            <h3 style={{ marginTop: 0 }}>Confirm Training</h3>
            <p>Are you sure you want to Start Training the Model?</p>
            <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>
              This will train the DSNN model with {epochs} epochs using the dataset at "{datasetPath}".
            </p>
            <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '24px' }}>
              <button 
                className="btn btn-secondary"
                onClick={() => setShowStartConfirm(false)}
              >
                Cancel
              </button>
              <button 
                className="btn btn-primary"
                onClick={handleStartTraining}
              >
                Start Training
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Stop Training Confirmation Modal */}
      {showStopConfirm && (
        <div className="modal-overlay">
          <div className="modal-content" style={{ maxWidth: '400px' }}>
            <h3 style={{ marginTop: 0 }}>Stop Training</h3>
            <p>Are you sure you want to Stop Training the Model?</p>
            <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>
              Stopping training will cancel the current session. No model will be saved.
            </p>
            <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '24px' }}>
              <button 
                className="btn btn-secondary"
                onClick={() => setShowStopConfirm(false)}
              >
                Cancel
              </button>
              <button 
                className="btn btn-danger"
                onClick={handleStopTraining}
              >
                Stop Training
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Full Screen Image Modal */}
      {expandedImage && (
        <div 
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0,0,0,0.9)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000
          }}
          onClick={() => setExpandedImage(null)}
        >
          <button
            className="btn btn-secondary"
            onClick={() => setExpandedImage(null)}
            style={{
              position: 'absolute',
              top: '16px',
              right: '16px'
            }}
          >
            <Minimize2 size={18} />
            Close
          </button>
          <img 
            src={expandedImage} 
            alt="Full screen" 
            style={{ 
              maxWidth: '90%', 
              maxHeight: '90%', 
              objectFit: 'contain' 
            }}
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}

      <style>{`
        .spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        
        .modal-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.5);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
        }
        
        .modal-content {
          background: var(--background-primary);
          border: 1px solid var(--border-color);
          border-radius: 12px;
          padding: 24px;
          max-width: 500px;
          width: 90%;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }
        
        .btn-danger {
          background: #ef4444;
          color: white;
          border: none;
          padding: 10px 20px;
          border-radius: 6px;
          cursor: pointer;
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
          transition: background 0.2s;
        }
        
        .btn-danger:hover {
          background: #dc2626;
        }
        
        .btn-danger:disabled {
          background: #fca5a5;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
};

export default ModelTraining;

