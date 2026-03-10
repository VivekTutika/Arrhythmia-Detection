import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  Activity, 
  FileText, 
  Play, 
  Settings,
  RefreshCw,
  CheckCircle,
  XCircle,
  Maximize2,
  Minimize2
} from 'lucide-react';
import { convertMitbih, trainModel, getTrainingStatus } from '../services/api';

const ModelTraining = ({ isLoading, setIsLoading }) => {
  const [activeTab, setActiveTab] = useState('training');
  const [converting, setConverting] = useState(false);
  const [conversionResult, setConversionResult] = useState(null);
  const [training, setTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [datasetPath, setDatasetPath] = useState('Dataset/edf');
  const [epochs, setEpochs] = useState(50);
  const [expandedImage, setExpandedImage] = useState(null);

  useEffect(() => {
    fetchTrainingStatus();
  }, []);

  const fetchTrainingStatus = async () => {
    try {
      const status = await getTrainingStatus();
      setTrainingStatus(status);
    } catch (error) {
      console.error('Error fetching training status:', error);
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

  const handleTrain = async () => {
    setTraining(true);
    setIsLoading(true);
    
    try {
      const result = await trainModel(datasetPath, epochs);
      console.log('Training started:', result);
      
      // Poll for status updates
      const pollInterval = setInterval(async () => {
        const status = await getTrainingStatus();
        setTrainingStatus(status);
        if (status.status === 'completed') {
          clearInterval(pollInterval);
          setTraining(false);
          setIsLoading(false);
        }
      }, 2000);
      
    } catch (error) {
      console.error('Training error:', error);
      setTraining(false);
      setIsLoading(false);
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
              <h3 className="card-title">Convert MIT-BIH Data</h3>
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
              
              <h3 style={{ margin: 0 }}>MIT-BIH Dataset Conversion</h3>
              <p style={{ color: 'var(--text-secondary)', textAlign: 'center', maxWidth: '450px' }}>
                Convert the MIT-BIH Arrhythmia Database files from .hea/.dat/.atr format to .edf/.qrs format for processing.
              </p>

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
                className="btn btn-primary"
                onClick={handleTrain}
                disabled={training || !datasetPath}
                style={{ minWidth: '200px' }}
              >
                {training ? (
                  <>
                    <RefreshCw size={18} className="spin" />
                    Training in Progress...
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
                <div className="processing-progress-text">Training model...</div>
              </div>
            )}

            {/* Training Results */}
            {trainingStatus && (
              <div>
                <h4 style={{ marginBottom: '16px' }}>Training Results</h4>
                
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
      `}</style>
    </div>
  );
};

export default ModelTraining;
