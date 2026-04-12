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
  Calendar,
  Image as ImageIcon,
  Download,
  UploadCloud
} from 'lucide-react';
import { convertMitbih, trainModel, getTrainingStatus, stopTraining, getECGVisualization, splitCSV } from '../services/api';

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

  // --- Dual-Mode State ---
  const [preprocessMode, setPreprocessMode] = useState('raw');   // 'raw' | 'csv'
  const [trainingMode, setTrainingMode] = useState('raw');        // 'raw' | 'csv'
  const [activeTrainingMode, setActiveTrainingMode] = useState('raw'); // mode of the last/running session
  const [csvSplitPath, setCsvSplitPath] = useState('');
  const [csvSplitResult, setCsvSplitResult] = useState(null);
  const [splitting, setSplitting] = useState(false);
  const [csvTrainPath, setCsvTrainPath] = useState('Dataset/CSV/splits');

  // --- New ECG Reader State ---
  const [ecgMode, setEcgMode] = useState('dataset');
  const [ecgRecord, setEcgRecord] = useState('100');
  const [ecgDuration, setEcgDuration] = useState(10);
  const [ecgEdfFile, setEcgEdfFile] = useState(null);
  const [ecgQrsFile, setEcgQrsFile] = useState(null);
  const [isRenderingEcg, setIsRenderingEcg] = useState(false);
  const [ecgImageUrl, setEcgImageUrl] = useState(null);
  const [ecgError, setEcgError] = useState(null);

  const handleGenerateECG = async () => {
    setIsRenderingEcg(true);
    setEcgError(null);
    setEcgImageUrl(null);
    try {
      const data = ecgMode === 'dataset' 
        ? { record_name: ecgRecord, duration: ecgDuration }
        : { edfFile: ecgEdfFile, qrsFile: ecgQrsFile, duration: ecgDuration };
        
      const result = await getECGVisualization(ecgMode, data);
      if (result.status === 'success' && result.image_url) {
        setEcgImageUrl(import.meta.env.DEV ? `http://localhost:5000${result.image_url}` : result.image_url);
      }
    } catch (error) {
      console.error(error);
      setEcgError(error.response?.data?.message || error.response?.data?.error || error.message || 'Error generating ECG');
    } finally {
      setIsRenderingEcg(false);
    }
  };

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

  const handleSplitCSV = async () => {
    if (!csvSplitPath.trim()) return;
    setSplitting(true);
    setCsvSplitResult(null);
    try {
      const result = await splitCSV(csvSplitPath.trim());
      setCsvSplitResult({ success: true, ...result });
    } catch (error) {
      setCsvSplitResult({
        success: false,
        error: error.response?.data?.error || error.message || 'Split failed'
      });
    } finally {
      setSplitting(false);
    }
  };

  const startPolling = (modeToUse) => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }

    pollIntervalRef.current = setInterval(async () => {
      try {
        const status = await getTrainingStatus(modeToUse);
        setTrainingStatus(status);

        if (status.status === 'completed' || status.status === 'stopped' || status.status === 'failed') {
          setTraining(false);
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }
      } catch (error) {
        console.error('Polling error:', error);
      }
    }, 5000);
  };

  const handleStartTraining = async () => {
    setShowStartConfirm(false);
    setTrainingStatus(null);
    setTraining(true);
    setActiveTrainingMode(trainingMode); // lock in the mode for this session
    try {
      const path = trainingMode === 'csv' ? csvTrainPath : datasetPath;
      const result = await trainModel(path, epochs, trainingMode);
      console.log('Training started:', result);
      startPolling(trainingMode);
    } catch (error) {
      console.error('Training error:', error);
      setTraining(false);
    }
  };

  const handleStopTraining = async () => {
    setShowStopConfirm(false);
    try {
      const result = await stopTraining(activeTrainingMode);
      console.log('Stop training result:', result);
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

  // ── When the user switches the Training Mode selector, load that mode's
  //    persisted status so results are always visible for the selected pipeline.
  useEffect(() => {
    if (training) return; // don't interrupt an active run
    let cancelled = false;
    getTrainingStatus(trainingMode).then(status => {
      if (!cancelled) setTrainingStatus(status);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [trainingMode]); // eslint-disable-line react-hooks/exhaustive-deps

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
          <button 
            className={`enhanced-tab ${activeTab === 'ecg-reader' ? 'active' : ''}`}
            onClick={() => setActiveTab('ecg-reader')}
          >
            <ImageIcon size={18} />
            <span>ECG Reader</span>
            <div className="tab-indicator"></div>
          </button>
        </div>
      </div>

      {/* Pre-Processing Tab */}
      {activeTab === 'preprocessing' && (
        <div className="card">
          <div className="card-header">
            <div>
              <h3 className="card-title">Pre-Processing</h3>
              <p className="card-subtitle">Select a pre-processing mode to prepare your dataset</p>
            </div>
          </div>

          {/* Mode Selector */}
          <div style={{ padding: '0 8px 24px' }}>
            <div style={{ display: 'flex', gap: '12px', marginBottom: '24px' }}>
              {[{ val: 'raw', label: 'Convert Raw MIT-BIH', icon: <RefreshCw size={18} /> },
                { val: 'csv', label: 'Split CSV Dataset', icon: <Database size={18} /> }]
                .map(({ val, label, icon }) => (
                  <label key={val} style={{
                    flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    gap: '10px', cursor: 'pointer', padding: '14px 16px', borderRadius: '10px',
                    border: preprocessMode === val ? '2px solid var(--primary-color)' : '2px solid var(--border-color)',
                    background: preprocessMode === val ? 'rgba(99,102,241,0.1)' : 'var(--background-secondary)',
                    color: preprocessMode === val ? 'var(--primary-color)' : 'var(--text-secondary)',
                    fontWeight: preprocessMode === val ? '600' : '500',
                    transition: 'all 0.2s',
                  }}>
                    <input type="radio" name="preprocessMode" value={val}
                      checked={preprocessMode === val} onChange={() => setPreprocessMode(val)}
                      style={{ display: 'none' }} />
                    {icon} {label}
                  </label>
                ))}
            </div>

            {/* RAW mode panel */}
            {preprocessMode === 'raw' && (
              <div style={{
                display: 'flex', flexDirection: 'column', alignItems: 'center',
                gap: '16px', padding: '32px',
                background: 'var(--background-secondary)', borderRadius: '12px'
              }}>
                <div style={{
                  width: '64px', height: '64px', borderRadius: '50%',
                  background: 'var(--primary-color)', display: 'flex',
                  alignItems: 'center', justifyContent: 'center'
                }}>
                  <RefreshCw size={32} color="white" />
                </div>
                <div className="processing-info">
                  <div className="processing-info-icon"><RefreshCw size={20} color="white" /></div>
                  <div className="processing-info-content">
                    <div className="processing-info-title">Estimated Processing Time</div>
                    <div className="processing-info-text">
                      Conversion typically takes 5–10 minutes depending on the number of files.
                      The process runs in the background and you'll be notified upon completion.
                    </div>
                  </div>
                </div>
                <button className="btn btn-primary" onClick={handleConvert}
                  disabled={converting} style={{ minWidth: '200px' }}>
                  {converting ? (<><RefreshCw size={18} className="spin" />Converting...</>)
                    : (<><Play size={18} />Convert MIT-BIH Data</>)}
                </button>
                {converting && (
                  <div className="processing-progress">
                    <div className="processing-progress-bar">
                      <div className="processing-progress-fill" style={{ width: '60%' }} />
                    </div>
                    <div className="processing-progress-text">Processing files...</div>
                  </div>
                )}
              </div>
            )}

            {/* RAW conversion result */}
            {preprocessMode === 'raw' && conversionResult && (
              <div style={{ marginTop: '24px' }}>
                {conversionResult.success ? (
                  <div className="status-banner status-success">
                    <CheckCircle size={24} style={{ color: '#10b981', flexShrink: 0 }} />
                    <div>
                      <h4 style={{ margin: '0 0 8px 0', color: '#10b981' }}>Conversion Successful!</h4>
                      <p style={{ margin: 0, color: 'var(--text-secondary)' }}>{conversionResult.message}</p>
                      {conversionResult.results && (
                        <div style={{ marginTop: '12px', display: 'flex', gap: '24px' }}>
                          <div><span style={{ fontWeight: 'bold' }}>EDF:</span> {conversionResult.results.edf.success} files</div>
                          <div><span style={{ fontWeight: 'bold' }}>QRS:</span> {conversionResult.results.qrs.success} files</div>
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

            {/* CSV Split mode panel */}
            {preprocessMode === 'csv' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div style={{
                  background: 'var(--background-secondary)', borderRadius: '12px', padding: '24px'
                }}>
                  <div className="processing-info" style={{ marginBottom: '20px' }}>
                    <div className="processing-info-icon"><Database size={20} color="white" /></div>
                    <div className="processing-info-content">
                      <div className="processing-info-title">Record-Based CSV Split</div>
                      <div className="processing-info-text">
                        Splits your CSV dataset into train/test sets using fixed test records
                        [101, 200, 207, 209, 213, 222, 228] to prevent data leakage.
                        Output saved to <code style={{ background: 'rgba(99,102,241,0.15)', padding: '2px 6px', borderRadius: '4px' }}>Dataset/CSV/splits/</code>
                      </div>
                    </div>
                  </div>
                  <div className="form-group">
                    <label className="form-label">CSV Dataset Path</label>
                    <input
                      type="text"
                      className="form-input"
                      value={csvSplitPath}
                      onChange={(e) => setCsvSplitPath(e.target.value)}
                      placeholder="e.g. Dataset/CSV/mitbih_features.csv"
                      disabled={splitting}
                    />
                    <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                      Absolute or relative path to your source CSV file containing a 'record' column.
                    </p>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'center' }}>
                    <button className="btn btn-primary" onClick={handleSplitCSV}
                      disabled={splitting || !csvSplitPath.trim()} style={{ minWidth: '200px' }}>
                      {splitting
                        ? (<><RefreshCw size={18} className="spin" />Splitting...</>)
                        : (<><Play size={18} />Split CSV Dataset</>)}
                    </button>
                  </div>
                </div>

                {csvSplitResult && (
                  <div>
                    {csvSplitResult.success ? (
                      <div className="status-banner status-success">
                        <CheckCircle size={24} style={{ color: '#10b981', flexShrink: 0 }} />
                        <div>
                          <h4 style={{ margin: '0 0 8px 0', color: '#10b981' }}>Split Successful!</h4>
                          <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap', fontSize: '13px', color: 'var(--text-secondary)' }}>
                            <span>Total rows: <strong>{csvSplitResult.total_rows?.toLocaleString()}</strong></span>
                            <span>Train rows: <strong style={{ color: '#10b981' }}>{csvSplitResult.train_rows?.toLocaleString()}</strong></span>
                            <span>Test rows: <strong style={{ color: '#6366f1' }}>{csvSplitResult.test_rows?.toLocaleString()}</strong></span>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="status-banner status-error">
                        <XCircle size={24} style={{ color: '#ef4444', flexShrink: 0 }} />
                        <div>
                          <h4 style={{ margin: '0 0 8px 0', color: '#ef4444' }}>Split Failed</h4>
                          <p style={{ margin: 0, color: 'var(--text-secondary)' }}>{csvSplitResult.error}</p>
                        </div>
                      </div>
                    )}
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
                <h3 className="card-title">Train Model</h3>
                <p className="card-subtitle">Configure a training mode and start training your arrhythmia detection model</p>
              </div>
            </div>

            <div style={{ padding: '8px' }}>
              {/* Training Mode Selector */}
              <div style={{ marginBottom: '24px' }}>
                <label className="form-label" style={{ marginBottom: '10px', display: 'block' }}>Training Mode</label>
                <div style={{ display: 'flex', gap: '12px' }}>
                  {[{ val: 'raw', label: 'Raw ECG (DSNN)', icon: <Activity size={18} /> },
                    { val: 'csv', label: 'CSV Dataset (MLP)', icon: <Database size={18} /> }]
                    .map(({ val, label, icon }) => (
                      <label key={val} style={{
                        flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
                        gap: '10px', cursor: training ? 'not-allowed' : 'pointer',
                        padding: '14px 16px', borderRadius: '10px',
                        border: trainingMode === val ? '2px solid var(--primary-color)' : '2px solid var(--border-color)',
                        background: trainingMode === val ? 'rgba(99,102,241,0.1)' : 'var(--background-secondary)',
                        color: trainingMode === val ? 'var(--primary-color)' : 'var(--text-secondary)',
                        fontWeight: trainingMode === val ? '600' : '500',
                        opacity: training ? 0.6 : 1,
                        transition: 'all 0.2s',
                      }}>
                        <input type="radio" name="trainingMode" value={val}
                          checked={trainingMode === val}
                          onChange={() => !training && setTrainingMode(val)}
                          style={{ display: 'none' }} />
                        {icon} {label}
                      </label>
                    ))}
                </div>
              </div>

              {/* Training Configuration — adapts to mode */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
                gap: '24px',
                marginBottom: '24px'
              }}>
                {trainingMode === 'raw' ? (
                  <div className="form-group">
                    <label className="form-label">EDF Dataset Path</label>
                    <input type="text" className="form-input"
                      value={datasetPath}
                      onChange={(e) => setDatasetPath(e.target.value)}
                      placeholder="Dataset/MIT-BIH"
                      disabled={training} />
                    <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                      Folder containing converted EDF + QRS files (run Pre-Processing first)
                    </p>
                  </div>
                ) : (
                  <div className="form-group">
                    <label className="form-label">CSV Splits Directory</label>
                    <input type="text" className="form-input"
                      value={csvTrainPath}
                      onChange={(e) => setCsvTrainPath(e.target.value)}
                      placeholder="Dataset/CSV/splits"
                      disabled={training} />
                    <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                      Folder containing <code>train.csv</code> + <code>test.csv</code> (run CSV Split in Pre-Processing first).
                      You can also provide a direct path to a <code>.csv</code> file to auto-split.
                    </p>
                  </div>
                )}

                <div className="form-group">
                  <label className="form-label">Number of Epochs</label>
                  <input type="number" className="form-input"
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value) || 50)}
                    min={1} max={5000} disabled={training} />
                  <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                    Training iterations (recommended: 50–200) (Min: 1, Max: 5000)
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
                      if (trainingMode === 'csv') {
                        // Fast Tabular Pipeline (~1-2 seconds per epoch total)
                        const secPerEpoch = 5.0;
                        const totalSec = Math.round(epochs * secPerEpoch) + 3; // +3s for loading
                        if (totalSec < 60) {
                          return `Training CSV Tabular data for ${epochs} epochs: ~${totalSec} seconds.`;
                        } else {
                          const mins = Math.ceil(totalSec / 60);
                          return `Training CSV Tabular data for ${epochs} epochs: ~${mins} minutes.`;
                        }
                      } else {
                        // Heavy Raw DSNN (~90-120 mins per 50 epochs -> ~126 seconds per epoch base)
                        const secPerEpoch = 126;
                        const totalSec = Math.round(epochs * secPerEpoch);
                        
                        if (totalSec < 3600) {
                          const mins = Math.ceil(totalSec / 60);
                          // Calculate a realistic 0.85x to 1.15x bound
                          const rangeLow = Math.max(1, Math.round(mins * 0.85));
                          const rangeHigh = Math.round(mins * 1.15);
                          return `Processing Raw EDF data for ${epochs} epochs: ~${rangeLow}-${rangeHigh} minutes.`;
                        } else {
                          const hrsLow = ((totalSec * 0.85) / 3600).toFixed(1);
                          const hrsHigh = ((totalSec * 1.15) / 3600).toFixed(1);
                          return `Processing Raw EDF data for ${epochs} epochs: ~${hrsLow}-${hrsHigh} hours.`;
                        }
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
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                      <span style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text-primary)' }}>
                        {trainingStatus.message || 'Initializing...'}
                      </span>
                      <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
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
                              <td style={{...tdStyle, color: '#10b981', fontWeight: h.train_acc === bestTrainAcc ? 700 : 400}}>
                                {h.train_acc}%
                              </td>
                              <td style={tdStyle}>{h.val_loss}</td>
                              <td style={{...tdStyle, color: '#6366f1', fontWeight: h.val_acc === bestValAcc ? 700 : 400}}>
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
          {hasImages() && (() => {
            const modeLabel = trainingStatus.mode === 'csv' ? 'CSV / MLP' : 'Raw ECG / DSNN';
            const IMAGE_TITLES = {
              'training_history_csv.png': 'MLP Training History (Loss & Accuracy)',
              'confusion_matrix_csv.png': 'MLP Confusion Matrix (Test Set)',
              'training_history.png':     'DSNN Training History (Loss & Accuracy)',
              'confusion_matrix.png':     'DSNN Confusion Matrix (Test Set)',
            };
            const getTitle = (url) => {
              const file = Object.keys(IMAGE_TITLES).find(k => url.includes(k));
              return file ? IMAGE_TITLES[file] : (url.includes('training') ? 'Training History' : 'Confusion Matrix');
            };
            return (
              <div className="card">
                <div className="card-header">
                  <div>
                    <h3 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <TrendingUp size={20} />
                      Training Visualizations
                      <span style={{
                        fontSize: '11px', fontWeight: '500', padding: '2px 8px',
                        borderRadius: '999px', marginLeft: '4px',
                        background: trainingStatus.mode === 'csv' ? 'rgba(16,185,129,0.15)' : 'rgba(99,102,241,0.15)',
                        color:      trainingStatus.mode === 'csv' ? '#10b981' : '#6366f1',
                      }}>{modeLabel}</span>
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
                      <div key={idx} style={{
                        background: 'var(--background-secondary)',
                        borderRadius: '8px', overflow: 'hidden'
                      }}>
                        <div style={{
                          padding: '12px', borderBottom: '1px solid var(--border-color)',
                          display: 'flex', alignItems: 'center', gap: '8px'
                        }}>
                          <span style={{
                            width: '8px', height: '8px', borderRadius: '50%', flexShrink: 0,
                            background: trainingStatus.mode === 'csv' ? '#10b981' : '#6366f1'
                          }} />
                          <span style={{ fontWeight: '500', fontSize: '13px' }}>{getTitle(img)}</span>
                        </div>
                        <div style={{ position: 'relative' }}>
                          <img
                            src={img}
                            alt={getTitle(img)}
                            style={{ width: '100%', height: 'auto', display: 'block' }}
                          />
                          <button
                            className="btn btn-sm btn-secondary"
                            onClick={() => setExpandedImage(img)}
                            style={{
                              position: 'absolute', top: '8px', right: '8px',
                              background: 'rgba(0,0,0,0.5)', color: 'white'
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
            );
          })()}

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

      {/* ECG Reader Tab */}
      {activeTab === 'ecg-reader' && (
        <div className="card">
          <div className="card-header">
            <div>
              <h3 className="card-title">ECG Reader</h3>
              <p className="card-subtitle">Visualize ECG signals with R-peaks from dataset or local uploads</p>
            </div>
          </div>
          
          <div style={{ padding: '24px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '24px', marginBottom: '24px' }}>
              
              <div className="form-group" style={{ gridColumn: '1 / -1' }}>
                <label className="form-label">Mode</label>
                <div style={{ display: 'flex', gap: '16px' }}>
                  <label style={{
                    flex: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '10px',
                    cursor: 'pointer',
                    padding: '16px',
                    borderRadius: '8px',
                    border: ecgMode === 'dataset' ? '2px solid var(--primary-color)' : '2px solid transparent',
                    background: ecgMode === 'dataset' ? 'rgba(99, 102, 241, 0.1)' : 'var(--background-secondary)',
                    color: ecgMode === 'dataset' ? 'var(--primary-color)' : 'var(--text-secondary)',
                    fontWeight: ecgMode === 'dataset' ? '600' : '500',
                    transition: 'all 0.2s',
                    boxShadow: ecgMode === 'dataset' ? '0 0 0 1px var(--primary-color)' : '0 0 0 1px var(--border-color)'
                  }}>
                    <input 
                      type="radio" 
                      name="ecgMode" 
                      value="dataset" 
                      checked={ecgMode === 'dataset'} 
                      onChange={() => setEcgMode('dataset')} 
                      style={{ display: 'none' }}
                    />
                    <Database size={20} />
                    Dataset Mode
                  </label>
                  <label style={{
                    flex: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '10px',
                    cursor: 'pointer',
                    padding: '16px',
                    borderRadius: '8px',
                    border: ecgMode === 'upload' ? '2px solid var(--primary-color)' : '2px solid transparent',
                    background: ecgMode === 'upload' ? 'rgba(99, 102, 241, 0.1)' : 'var(--background-secondary)',
                    color: ecgMode === 'upload' ? 'var(--primary-color)' : 'var(--text-secondary)',
                    fontWeight: ecgMode === 'upload' ? '600' : '500',
                    transition: 'all 0.2s',
                    boxShadow: ecgMode === 'upload' ? '0 0 0 1px var(--primary-color)' : '0 0 0 1px var(--border-color)'
                  }}>
                    <input 
                      type="radio" 
                      name="ecgMode" 
                      value="upload" 
                      checked={ecgMode === 'upload'} 
                      onChange={() => setEcgMode('upload')} 
                      style={{ display: 'none' }}
                    />
                    <UploadCloud size={20} />
                    Upload Mode
                  </label>
                </div>
              </div>

              {ecgMode === 'dataset' ? (
                <div className="form-group">
                  <label className="form-label">MIT-BIH Record</label>
                  <select 
                    className="form-input" 
                    value={ecgRecord} 
                    onChange={(e) => setEcgRecord(e.target.value)}
                  >
                    {['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115', '116', '117', '118', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234'].map(rec => (
                      <option key={rec} value={rec}>{rec}</option>
                    ))}
                  </select>
                </div>
              ) : (
                <div className="form-group" style={{ gridColumn: '1 / -1' }}>
                  <label className="form-label">Upload Files</label>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px' }}>
                    
                    {/* EDF File Upload */}
                    <div 
                      className={`upload-area ${ecgEdfFile ? 'has-file' : ''}`}
                      style={{ 
                        padding: '24px 16px', 
                        minHeight: '120px', 
                        display: 'flex', 
                        flexDirection: 'column', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        gap: '12px', 
                        border: ecgEdfFile ? '2px dashed var(--primary-color)' : '2px dashed var(--border-color)',
                        background: ecgEdfFile ? 'rgba(99, 102, 241, 0.05)' : 'var(--background-secondary)',
                        borderRadius: '12px',
                        cursor: 'pointer',
                        transition: 'all 0.2s'
                      }}
                      onClick={() => document.getElementById('ecg-edf-input').click()}
                    >
                      <input 
                        type="file" 
                        id="ecg-edf-input"
                        accept=".edf" 
                        style={{ display: 'none' }}
                        onChange={(e) => setEcgEdfFile(e.target.files[0])} 
                      />
                      <UploadCloud size={32} style={{ color: ecgEdfFile ? 'var(--primary-color)' : 'var(--text-secondary)' }} />
                      <div style={{ textAlign: 'center' }}>
                        <span style={{ display: 'block', fontWeight: '500', color: ecgEdfFile ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                          {ecgEdfFile ? ecgEdfFile.name : 'Select EDF File'}
                        </span>
                        <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>(Required)</span>
                      </div>
                    </div>

                    {/* QRS File Upload */}
                    <div 
                      className={`upload-area ${ecgQrsFile ? 'has-file' : ''}`}
                      style={{ 
                        padding: '24px 16px', 
                        minHeight: '120px', 
                        display: 'flex', 
                        flexDirection: 'column', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        gap: '12px', 
                        border: ecgQrsFile ? '2px dashed var(--primary-color)' : '2px dashed var(--border-color)',
                        background: ecgQrsFile ? 'rgba(99, 102, 241, 0.05)' : 'var(--background-secondary)',
                        borderRadius: '12px',
                        cursor: 'pointer',
                        transition: 'all 0.2s'
                      }}
                      onClick={() => document.getElementById('ecg-qrs-input').click()}
                    >
                      <input 
                        type="file" 
                        id="ecg-qrs-input"
                        accept=".qrs" 
                        style={{ display: 'none' }}
                        onChange={(e) => setEcgQrsFile(e.target.files[0])} 
                      />
                      <UploadCloud size={32} style={{ color: ecgQrsFile ? 'var(--primary-color)' : 'var(--text-secondary)' }} />
                      <div style={{ textAlign: 'center' }}>
                        <span style={{ display: 'block', fontWeight: '500', color: ecgQrsFile ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                          {ecgQrsFile ? ecgQrsFile.name : 'Select QRS File'}
                        </span>
                        <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>(Optional)</span>
                      </div>
                    </div>

                  </div>
                </div>
              )}

              <div className="form-group">
                <label className="form-label">Duration (seconds)</label>
                <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
                  <input 
                    type="number" 
                    className="form-input" 
                    value={ecgDuration} 
                    onChange={(e) => setEcgDuration(parseInt(e.target.value) || 10)} 
                    min="1" 
                    max="60" 
                    style={{ flex: 1 }}
                  />
                  <button 
                    className="btn btn-secondary" 
                    onClick={() => {
                      setEcgMode('dataset');
                      setEcgRecord('100');
                      setEcgDuration(10);
                      setEcgEdfFile(null);
                      setEcgQrsFile(null);
                      setEcgImageUrl(null);
                      setEcgError(null);
                    }}
                    title="Clear Form"
                    disabled={isRenderingEcg}
                    style={{ height: '42px', width: '42px', padding: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, borderRadius: '8px' }}
                  >
                    <RefreshCw size={18} />
                  </button>
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '24px' }}>
              <button 
                className="btn btn-primary" 
                onClick={handleGenerateECG} 
                disabled={isRenderingEcg || (ecgMode === 'upload' && !ecgEdfFile)}
                style={{ minWidth: '200px' }}
              >
                {isRenderingEcg ? (
                  <>
                    <RefreshCw size={18} className="spin" />
                    Rendering ECG...
                  </>
                ) : (
                  <>
                    <ImageIcon size={18} />
                    Generate ECG
                  </>
                )}
              </button>
            </div>

            {ecgError && (
               <div className="status-banner status-error" style={{ marginBottom: '24px' }}>
                 <XCircle size={24} style={{ color: '#ef4444', flexShrink: 0 }} />
                 <div>
                   <h4 style={{ margin: '0 0 8px 0', color: '#ef4444' }}>Error Generating ECG</h4>
                   <p style={{ margin: 0, color: 'var(--text-secondary)' }}>
                     {ecgError}
                   </p>
                 </div>
               </div>
            )}

            {ecgImageUrl && (
              <div style={{ background: 'var(--background-secondary)', padding: '16px', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                  <h4 style={{ margin: 0 }}>Generated Visualization</h4>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <a 
                      href={`${ecgImageUrl}?download=1`}
                      className="btn btn-sm btn-secondary"
                      download={`ecg_${ecgMode === 'dataset' ? ecgRecord : 'uploaded'}.png`}
                    >
                      <Download size={14} /> Download
                    </a>
                    <button 
                      className="btn btn-sm btn-secondary" 
                      onClick={() => setExpandedImage(ecgImageUrl)}
                    >
                      <Maximize2 size={14} /> Full Screen
                    </button>
                  </div>
                </div>
                <div style={{ background: 'white', borderRadius: '8px', overflow: 'hidden', display: 'flex', justifyContent: 'center' }}>
                  <img 
                    src={ecgImageUrl} 
                    alt="ECG Signal" 
                    style={{ maxWidth: '100%', height: 'auto', display: 'block', cursor: 'pointer' }}
                    onClick={() => setExpandedImage(ecgImageUrl)}
                  />
                </div>
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
            <p>Are you sure you want to start training the model?</p>
            <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>
              {trainingMode === 'csv'
                ? <>Will train the <strong>MLP (CSV)</strong> model for {epochs} epochs using splits at <em>"{csvTrainPath}"</em>.</>
                : <>Will train the <strong>DSNN (Raw ECG)</strong> model for {epochs} epochs using dataset at <em>"{datasetPath}"</em>.</>
              }
            </p>
            <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '24px' }}>
              <button className="btn btn-secondary" onClick={() => setShowStartConfirm(false)}>Cancel</button>
              <button className="btn btn-primary" onClick={handleStartTraining}>Start Training</button>
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
