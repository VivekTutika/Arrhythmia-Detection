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
  Minimize2,
  AlertCircle,
  TrendingUp,
  BarChart3,
  Database,
  Clock,
  Calendar
} from 'lucide-react';
import { convertMitbih, trainModel, getTrainingStatus, stopTraining } from '../services/api';

const ModelTraining = ({ setIsLoading }) => {
  const [activeTab, setActiveTab] = useState('training');
  const [converting, setConverting] = useState(false);
  const [conversionResult, setConversionResult] = useState(null);
  const [training, setTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const getInitialDataPath = () => {
    try {
      const saved = localStorage.getItem('appSettings');
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed.dataPath) return parsed.dataPath;
      }
    } catch (e) { console.error(e); }
    return 'Dataset/MIT-BIH';
  };

  const [datasetPath, setDatasetPath] = useState(getInitialDataPath());
  const [epochs, setEpochs] = useState(50);
  const [expandedImage, setExpandedImage] = useState(null);
  const [showStartConfirm, setShowStartConfirm] = useState(false);
  const [showStopConfirm, setShowStopConfirm] = useState(false);
  const pollIntervalRef = useRef(null);

  useEffect(() => {
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

      if (status.status === 'running') {
        setTraining(true);
        // Start polling if training is running (e.g., page reload during training)
        startPolling();
      } else {
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
      try {
        const status = await getTrainingStatus();
        setTrainingStatus(status);

        if (status.status === 'completed' || status.status === 'stopped' || status.status === 'failed') {
          setTraining(false);
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }
      } catch (error) {
        console.error('Polling error:', error);
      }
    }, 60000);
  };

  const handleStartTraining = async () => {
    setShowStartConfirm(false);
    setTrainingStatus(null);
    setTraining(true);

    try {
      const result = await trainModel(datasetPath, epochs);
      console.log('Training started:', result);
      startPolling();
    } catch (error) {
      console.error('Training error:', error);
      setTraining(false);
    }
  };

  const handleStopTraining = async () => {
    setShowStopConfirm(false);

    try {
      const result = await stopTraining();
      console.log('Stop training result:', result);
      // Polling will handle updating the status
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

  // Helper: get the latest metrics from the current/latest training
  const getLatestMetrics = () => {
    if (!trainingStatus?.metrics) return null;
    return trainingStatus.metrics;
  };

  // Helper: check if images exist
  const hasImages = () => {
    return trainingStatus?.image_files && trainingStatus.image_files.length > 0;
  };

  // Helper: check if models exist
  const hasModels = () => {
    return trainingStatus?.model_exists;
  };

  return (
    <div className="model-training-page">
      {/* Tab Navigation */}
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

              <div className="processing-info">
                <div className="processing-info-icon">
                  <RefreshCw size={20} color="white" />
                </div>
                <div className="processing-info-content">
                  <div className="processing-info-title">Estimated Processing Time</div>
                  <div className="processing-info-text">
                    Conversion typically takes 5-10 minutes depending on the number of files.
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

              {converting && (
                <div className="processing-progress">
                  <div className="processing-progress-bar">
                    <div className="processing-progress-fill" style={{ width: '60%' }}></div>
                  </div>
                  <div className="processing-progress-text">Processing files...</div>
                </div>
              )}
            </div>

            {conversionResult && (
              <div style={{ marginTop: '24px' }}>
                {conversionResult.success ? (
                  <div className="status-banner status-success">
                    <CheckCircle size={24} style={{ color: '#10b981', flexShrink: 0 }} />
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
                  <div className="status-banner status-error">
                    <XCircle size={24} style={{ color: '#ef4444', flexShrink: 0 }} />
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
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

          {/* Existing Model / Image Info Banner */}
          {!training && trainingStatus && trainingStatus.status !== 'running' && (
            <>
              {hasModels() && (
                <div className="status-banner status-info">
                  <Database size={22} style={{ color: '#6366f1', flexShrink: 0 }} />
                  <div>
                    <h4 style={{ margin: '0 0 4px 0', color: '#6366f1' }}>Trained Models Available</h4>
                    <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '13px' }}>
                      {trainingStatus.model_files && Object.entries(trainingStatus.model_files)
                        .filter(([_, exists]) => exists)
                        .map(([name]) => name)
                        .join(', ')}
                      {' — '}You can retrain the model by starting a new training session below.
                    </p>
                  </div>
                </div>
              )}
            </>
          )}

          {/* Training Configuration Card */}
          <div className="card">
            <div className="card-header">
              <div>
                <h3 className="card-title">Train DSNN Model</h3>
                <p className="card-subtitle">Configure and train the Deep Spiking Neural Network for Arrhythmia Detection</p>
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
                    placeholder="Dataset/MIT-BIH"
                    disabled={training}
                  />
                  <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                    Path to the folder containing converted EDF and QRS files (run Pre-Processing first)
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

              <div className="processing-info">
                <div className="processing-info-icon">
                  <Activity size={20} color="white" />
                </div>
                <div className="processing-info-content">
                  <div className="processing-info-title">Estimated Processing Time</div>
                  <div className="processing-info-text">
                    {(() => {
                      const trainFiles = 40;
                      const secPerEpochPerFile = 0.75;
                      const totalSec = Math.round(trainFiles * epochs * secPerEpochPerFile);
                      const dataLoadOverhead = Math.round(trainFiles * 2);
                      const totalWithOverhead = totalSec + dataLoadOverhead;

                      if (totalWithOverhead < 60) {
                        return `Training ${trainFiles} files for ${epochs} epochs: ~${totalWithOverhead} seconds.`;
                      } else if (totalWithOverhead < 3600) {
                        const mins = Math.ceil(totalWithOverhead / 60);
                        return `Training ${trainFiles} files for ${epochs} epochs: ~${mins} minutes.`;
                      } else {
                        const hrs = (totalWithOverhead / 3600).toFixed(1);
                        return `Training ${trainFiles} files for ${epochs} epochs: ~${hrs} hours.`;
                      }
                    })()}
                    {' '}Time may vary based on your device performance (CPU vs GPU).
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

              {/* Live Training Progress */}
              {training && trainingStatus && (
                <div className="training-progress-section">
                  <div className="processing-progress">
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text-primary)', marginRight: '8px' }}>
                        {`${trainingStatus.message} |` || 'Initializing...'}
                      </span>
                      <span style={{ fontSize: '13px', color: 'var(--text-primary)' }}>
                        {trainingStatus.progress || 0}%
                      </span>
                    </div>
                    <div className="processing-progress-bar">
                      <div
                        className="processing-progress-fill"
                        style={{
                          width: `${trainingStatus.progress || 0}%`,
                          transition: 'width 0.5s ease'
                        }}
                      ></div>
                    </div>
                    <div className="processing-progress-text" style={{ marginTop: '6px' }}>
                      {trainingStatus.current_epoch > 0
                        ? `Epoch ${trainingStatus.current_epoch}/${trainingStatus.epochs} — Click "Stop Training" to cancel`
                        : 'Loading data and initializing model... This may take a minute.'
                      }
                    </div>
                  </div>
                </div>
              )}

              {/* Status Messages (completed / stopped / failed) */}
              {!training && trainingStatus && trainingStatus.status !== 'not_started' && (
                <div style={{ marginBottom: '24px' }}>
                  {trainingStatus.status === 'completed' && (
                    <div className="status-banner status-success">
                      <CheckCircle size={24} style={{ color: '#10b981', flexShrink: 0 }} />
                      <div style={{ flex: 1 }}>
                        <h4 style={{ margin: '0 0 4px 0', color: '#10b981' }}>Training Completed Successfully!</h4>
                        <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '13px' }}>
                          The model has been trained on clinical MIT-BIH data. Evaluation results are available below.
                        </p>
                      </div>
                      <div style={{ textAlign: 'right', fontSize: '11px', color: 'var(--text-secondary)' }}>
                        <div>Completed at: {new Date(trainingStatus.end_time).toLocaleTimeString()}</div>
                        <div>Date: {new Date(trainingStatus.end_time).toLocaleDateString()}</div>
                      </div>
                    </div>
                  )}

                  {trainingStatus.status === 'stopped' && (
                    <div className="status-banner status-warning">
                      <AlertCircle size={24} style={{ color: '#f59e0b', flexShrink: 0 }} />
                      <div style={{ flex: 1 }}>
                        <h4 style={{ margin: '0 0 4px 0', color: '#f59e0b' }}>Training Stopped</h4>
                        <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '13px' }}>
                          Training was stopped by user. Showing partial metrics and visualizations below.
                        </p>
                      </div>
                    </div>
                  )}

                  {trainingStatus.status === 'failed' && (
                    <div className="status-banner status-error">
                      <XCircle size={24} style={{ color: '#ef4444', flexShrink: 0 }} />
                      <div style={{ flex: 1 }}>
                        <h4 style={{ margin: '0 0 4px 0', color: '#ef4444' }}>Training Failed</h4>
                        <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '13px' }}>
                          {trainingStatus.error || 'Check logs for details.'}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Past Training Info Indicator */}
                  {!training && (
                    <div style={{
                      marginTop: '12px',
                      display: 'flex',
                      gap: '16px',
                      fontSize: '12px',
                      color: 'var(--text-secondary)',
                      padding: '0 8px'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Clock size={14} />
                        Duration: {(() => {
                          if (!trainingStatus.start_time || !trainingStatus.end_time) return 'N/A';
                          const start = new Date(trainingStatus.start_time);
                          const end = new Date(trainingStatus.end_time);
                          if (isNaN(start.getTime()) || isNaN(end.getTime())) return 'N/A';
                          const diff = Math.floor((end - start) / 1000);
                          const mins = Math.floor(diff / 60);
                          const secs = diff % 60;
                          return `${mins}m ${secs}s`;
                        })()}
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Calendar size={14} />
                        Completed: {new Date(trainingStatus.end_time).toLocaleDateString()}
                      </div>
                    </div>
                  )}
                </div>
              )}

            </div>
          </div>

          {/* Metrics Section */}
          {(() => {
            const metrics = getLatestMetrics();
            const history = metrics?.history;
            const showMetrics = history && history.length > 0;

            if (!showMetrics) return null;

            const latestEpoch = history[history.length - 1];
            const bestValAcc = Math.max(...history.map(h => h.val_acc));
            const bestTrainAcc = Math.max(...history.map(h => h.train_acc));
            const lowestValLoss = Math.min(...history.map(h => h.val_loss));

            return (
              <div className="card">
                <div className="card-header">
                  <div>
                    <h3 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <BarChart3 size={20} />
                      Training Metrics
                    </h3>
                    <p className="card-subtitle">
                      {training
                        ? `Live metrics — Epoch ${latestEpoch.epoch}`
                        : `Final results — ${history.length} epochs completed`}
                    </p>
                  </div>
                </div>
                <div style={{ padding: '8px' }}>
                  {/* Summary Cards */}
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
                    gap: '12px',
                    marginBottom: '20px'
                  }}>
                    <div className="metric-card">
                      <div className="metric-label">Current Epoch</div>
                      <div className="metric-value">{latestEpoch.epoch} <span className="metric-unit">/ {trainingStatus?.epochs || history.length}</span></div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-label">Train Accuracy</div>
                      <div className="metric-value" style={{ color: '#10b981' }}>{latestEpoch.train_acc}%</div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-label">Val Accuracy</div>
                      <div className="metric-value" style={{ color: '#6366f1' }}>{latestEpoch.val_acc}%</div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-label">Train Loss</div>
                      <div className="metric-value" style={{ color: '#f59e0b' }}>{latestEpoch.train_loss}</div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-label">Val Loss</div>
                      <div className="metric-value" style={{ color: '#ef4444' }}>{latestEpoch.val_loss}</div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-label">Best Val Accuracy</div>
                      <div className="metric-value" style={{ color: '#10b981' }}>{bestValAcc}%</div>
                    </div>
                  </div>

                  {/* Evaluation Metrics (Show only when completed) */}
                  {metrics?.evaluation && (
                    <div style={{ marginBottom: '24px' }}>
                      <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '12px', color: 'var(--text-primary)' }}>
                        Final Evaluation on Unseen Test Records
                      </div>
                      <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                        gap: '12px'
                      }}>
                        <div style={{ background: 'rgba(16, 185, 129, 0.1)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                          <div style={{ fontSize: '12px', color: '#10b981', fontWeight: 600 }}>OVERALL ACCURACY</div>
                          <div style={{ fontSize: '24px', fontWeight: 700, color: '#10b981' }}>{(metrics.evaluation.accuracy * 100).toFixed(1)}%</div>
                        </div>
                        <div style={{ background: 'var(--background-secondary)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                          <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>PRECISION</div>
                          <div style={{ fontSize: '20px', fontWeight: 600 }}>{(metrics.evaluation.precision * 100).toFixed(1)}%</div>
                        </div>
                        <div style={{ background: 'var(--background-secondary)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                          <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>RECALL</div>
                          <div style={{ fontSize: '20px', fontWeight: 600 }}>{(metrics.evaluation.recall * 100).toFixed(1)}%</div>
                        </div>
                        <div style={{ background: 'var(--background-secondary)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                          <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>F1-SCORE</div>
                          <div style={{ fontSize: '20px', fontWeight: 600 }}>{(metrics.evaluation.f1 * 100).toFixed(1)}%</div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Epoch History Table */}
                  {history.length > 1 && (
                    <div style={{ maxHeight: '300px', overflowY: 'auto', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                        <thead>
                          <tr style={{ background: 'var(--background-secondary)', position: 'sticky', top: 0 }}>
                            <th style={thStyle}>Epoch</th>
                            <th style={thStyle}>Train Loss</th>
                            <th style={thStyle}>Train Acc</th>
                            <th style={thStyle}>Val Loss</th>
                            <th style={thStyle}>Val Acc</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[...history].reverse().map((h, idx) => (
                            <tr key={h.epoch} style={{ background: idx % 2 === 0 ? 'transparent' : 'var(--background-secondary)' }}>
                              <td style={tdStyle}>{h.epoch}</td>
                              <td style={tdStyle}>{h.train_loss}</td>
                              <td style={{ ...tdStyle, color: '#10b981', fontWeight: h.train_acc === bestTrainAcc ? 700 : 400 }}>
                                {h.train_acc}%
                              </td>
                              <td style={tdStyle}>{h.val_loss}</td>
                              <td style={{ ...tdStyle, color: '#6366f1', fontWeight: h.val_acc === bestValAcc ? 700 : 400 }}>
                                {h.val_acc}%
                                {h.val_acc === bestValAcc && <span style={{ marginLeft: 4, fontSize: '11px' }}>⭐</span>}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>
            );
          })()}

          {/* Output Images Section */}
          {hasImages() && (
            <div className="card">
              <div className="card-header">
                <div>
                  <h3 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <TrendingUp size={20} />
                    Training Visualizations
                  </h3>
                  <p className="card-subtitle">Model training history and evaluation results</p>
                </div>
              </div>
              <div style={{ padding: '8px' }}>
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
                          alt={img.includes('training') ? 'Training History' : 'Confusion Matrix'}
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
              </div>
            </div>
          )}

          {/* No Results Yet */}
          {!hasImages() && !training && trainingStatus && trainingStatus.status === 'not_started' && !hasModels() && (
            <div className="card">
              <div style={{
                padding: '48px 24px',
                textAlign: 'center',
                color: 'var(--text-secondary)'
              }}>
                <Activity size={48} style={{ opacity: 0.3, marginBottom: '16px' }} />
                <p style={{ fontSize: '15px', margin: 0 }}>
                  No training results yet. Configure the parameters above and start training to see results.
                </p>
              </div>
            </div>
          )}
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

        .status-banner {
          padding: 16px;
          border-radius: 10px;
          display: flex;
          align-items: flex-start;
          gap: 12px;
          margin-bottom: 4px;
        }
        .status-banner.status-success {
          background: rgba(16, 185, 129, 0.08);
          border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .status-banner.status-warning {
          background: rgba(245, 158, 11, 0.08);
          border: 1px solid rgba(245, 158, 11, 0.3);
        }
        .status-banner.status-error {
          background: rgba(239, 68, 68, 0.08);
          border: 1px solid rgba(239, 68, 68, 0.3);
        }
        .status-banner.status-info {
          background: rgba(99, 102, 241, 0.08);
          border: 1px solid rgba(99, 102, 241, 0.3);
        }

        .training-progress-section {
          background: var(--background-secondary);
          border-radius: 10px;
          padding: 20px;
          margin-bottom: 16px;
        }

        .metric-card {
          background: var(--background-secondary);
          border: 1px solid var(--border-color);
          border-radius: 10px;
          padding: 16px;
          text-align: center;
        }
        .metric-label {
          font-size: 12px;
          color: var(--text-secondary);
          text-transform: uppercase;
          letter-spacing: 0.5px;
          margin-bottom: 6px;
        }
        .metric-value {
          font-size: 22px;
          font-weight: 700;
          color: var(--text-primary);
        }
        .metric-unit {
          font-size: 13px;
          font-weight: 400;
          color: var(--text-secondary);
        }
      `}</style>
    </div>
  );
};

// Table styles
const thStyle = {
  padding: '10px 14px',
  textAlign: 'left',
  fontWeight: 600,
  color: 'var(--text-secondary)',
  borderBottom: '1px solid var(--border-color)',
  fontSize: '12px',
  textTransform: 'uppercase',
  letterSpacing: '0.5px'
};

const tdStyle = {
  padding: '8px 14px',
  borderBottom: '1px solid var(--border-color)',
  color: 'var(--text-primary)'
};

export default ModelTraining;
