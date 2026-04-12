import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload as UploadIcon, FileAudio, CheckCircle, User, Calendar, FileSpreadsheet, Activity, Database } from 'lucide-react';
import defaultToast from 'react-hot-toast';

const toast = {
  success: (msg) => {
    try {
      const stored = localStorage.getItem('appSettings');
      if (!stored || JSON.parse(stored).notifications) defaultToast.success(msg);
    } catch { defaultToast.success(msg); }
  },
  error: (msg) => {
    try {
      const stored = localStorage.getItem('appSettings');
      if (!stored || JSON.parse(stored).notifications) defaultToast.error(msg);
    } catch { defaultToast.error(msg); }
  }
};
import { analyzeECG, analyzeCSV } from '../services/api';

const Upload = () => {
  const navigate = useNavigate();

  // --- Analysis Mode ---
  const [analysisMode, setAnalysisMode] = useState('csv'); // 'edf' | 'csv'

  // --- EDF mode state (existing) ---
  const [isDragging, setIsDragging] = useState(false);
  const [files, setFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({});

  // --- CSV mode state (new) ---
  const [csvFile, setCsvFile] = useState(null);
  const [isCsvAnalyzing, setIsCsvAnalyzing] = useState(false);
  const [csvDragOver, setCsvDragOver] = useState(false);

  // --- Patient information (shared) ---
  const [patientName, setPatientName] = useState('');
  const [patientAge, setPatientAge] = useState('');
  const [showPatientForm, setShowPatientForm] = useState(false);

  // ===========================================================
  // EDF mode handlers (existing — unchanged)
  // ===========================================================
  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFiles = Array.from(e.dataTransfer.files).filter(
      file => file.name.endsWith('.edf') || file.name.endsWith('.qrs')
    );
    if (droppedFiles.length > 0) {
      setFiles(prev => [...prev, ...droppedFiles]);
      toast.success(`${droppedFiles.length} file(s) added`);
    } else {
      toast.error('Please upload EDF or QRS files only');
    }
  }, []);

  const handleFileSelect = (e) => {
    const selectedFiles = Array.from(e.target.files).filter(
      file => file.name.endsWith('.edf') || file.name.endsWith('.qrs')
    );
    if (selectedFiles.length > 0) {
      setFiles(prev => [...prev, ...selectedFiles]);
      toast.success(`${selectedFiles.length} file(s) added`);
    }
  };

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      toast.error('Please select at least one file');
      return;
    }
    if (!showPatientForm && !patientName) {
      setShowPatientForm(true);
      toast.error('Please enter patient information');
      return;
    }

    setIsUploading(true);
    const uploadedIds = [];
    let firstResultData = null;

    const fileGroups = {};
    files.forEach(f => {
      const baseName = f.name.substring(0, f.name.lastIndexOf('.'));
      const ext = f.name.substring(f.name.lastIndexOf('.') + 1).toLowerCase();
      if (!fileGroups[baseName]) fileGroups[baseName] = {};
      fileGroups[baseName][ext] = f;
    });

    let index = 0;
    for (const [baseName, group] of Object.entries(fileGroups)) {
      if (!group.edf) {
        toast.error(`Missing .edf file for ${baseName}`);
        continue;
      }

      const edfFile = group.edf;
      const qrsFile = group.qrs || null;

      setUploadProgress(prev => ({ ...prev, [edfFile.name]: 'uploading' }));
      if (qrsFile) setUploadProgress(prev => ({ ...prev, [qrsFile.name]: 'uploading' }));

      try {
        const patientInfo = {
          name: patientName || 'Anonymous',
          age: patientAge || 'N/A',
          id: `patient_${Date.now()}_${index}`
        };
        const response = await analyzeECG(edfFile, qrsFile, patientInfo);
        uploadedIds.push(response.id);
        if (index === 0 && response.result_data) firstResultData = response.result_data;

        setUploadProgress(prev => ({ ...prev, [edfFile.name]: 'completed' }));
        if (qrsFile) setUploadProgress(prev => ({ ...prev, [qrsFile.name]: 'completed' }));
        toast.success(`Analysis complete for ${baseName}`);
      } catch (error) {
        setUploadProgress(prev => ({ ...prev, [edfFile.name]: 'error' }));
        if (qrsFile) setUploadProgress(prev => ({ ...prev, [qrsFile.name]: 'error' }));
        toast.error(`Failed to analyze ${baseName}`);
      }
      index++;
    }

    setIsUploading(false);
    if (uploadedIds.length > 0) {
      navigate(`/results/${uploadedIds[0]}`, { state: { resultData: firstResultData } });
    }
  };

  // ===========================================================
  // CSV mode handlers (new)
  // ===========================================================
  const handleCsvDragOver = useCallback((e) => {
    e.preventDefault();
    setCsvDragOver(true);
  }, []);

  const handleCsvDragLeave = useCallback((e) => {
    e.preventDefault();
    setCsvDragOver(false);
  }, []);

  const handleCsvDrop = useCallback((e) => {
    e.preventDefault();
    setCsvDragOver(false);
    const dropped = Array.from(e.dataTransfer.files).find(f => f.name.endsWith('.csv'));
    if (dropped) {
      setCsvFile(dropped);
      toast.success(`${dropped.name} ready for analysis`);
    } else {
      toast.error('Please upload a .csv file');
    }
  }, []);

  const handleCsvFileSelect = (e) => {
    const f = e.target.files[0];
    if (f && f.name.endsWith('.csv')) {
      setCsvFile(f);
      toast.success(`${f.name} ready for analysis`);
    }
  };

  const handleCsvAnalyze = async () => {
    if (!csvFile) {
      toast.error('Please select a CSV file first');
      return;
    }
    if (!showPatientForm && !patientName) {
      setShowPatientForm(true);
      toast.error('Please enter patient information');
      return;
    }

    setIsCsvAnalyzing(true);
    try {
      const patientInfo = {
        name: patientName || 'Anonymous',
        age: patientAge || 'N/A',
        id: `patient_${Date.now()}`
      };
      const response = await analyzeCSV(csvFile, patientInfo);
      toast.success('CSV analysis complete!');
      navigate(`/results/${response.id}`, { state: { resultData: response.result_data } });
    } catch (error) {
      toast.error(error.response?.data?.error || error.message || 'CSV analysis failed');
    } finally {
      setIsCsvAnalyzing(false);
    }
  };

  // ===========================================================
  // Render
  // ===========================================================
  return (
    <div className="upload-page">
      <div className="page-header">
        <div>
          <h1>ECG Analysis</h1>
          <p className="text-secondary">Upload ECG data for arrhythmia detection</p>
        </div>
      </div>

      {/* ── Analysis Mode Selector ── */}
      <div className="card" style={{ marginBottom: '24px' }}>
        <div className="card-header">
          <div>
            <h3 className="card-title">Analysis Mode</h3>
            <p className="card-subtitle">Choose the input format for your ECG data</p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '12px', padding: '0 0 8px' }}>
          {[
            { val: 'edf', label: 'EDF + QRS', sub: 'Raw ECG (DSNN)', icon: <Activity size={20} /> },
            { val: 'csv', label: 'CSV File', sub: 'Feature-based (MLP)', icon: <Database size={20} /> },
          ].map(({ val, label, sub, icon }) => (
            <label key={val} style={{
              flex: 1, display: 'flex', alignItems: 'center', gap: '14px',
              cursor: 'pointer', padding: '16px 20px', borderRadius: '12px',
              border: analysisMode === val ? '2px solid var(--primary-color)' : '2px solid var(--border-color)',
              background: analysisMode === val ? 'rgba(99,102,241,0.08)' : 'var(--background-secondary)',
              transition: 'all 0.2s',
            }}>
              <input type="radio" name="analysisMode" value={val}
                checked={analysisMode === val} onChange={() => setAnalysisMode(val)}
                style={{ display: 'none' }} />
              <span style={{ color: analysisMode === val ? 'var(--primary-color)' : 'var(--text-secondary)' }}>
                {icon}
              </span>
              <div>
                <div style={{
                  fontWeight: '600', fontSize: '15px',
                  color: analysisMode === val ? 'var(--primary-color)' : 'var(--text-primary)',
                }}>{label}</div>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{sub}</div>
              </div>
              {analysisMode === val && (
                <CheckCircle size={18} style={{ color: 'var(--primary-color)', marginLeft: 'auto' }} />
              )}
            </label>
          ))}
        </div>
      </div>

      {/* ── Patient Information (shared) ── */}
      <div className="card" style={{ marginBottom: '24px' }}>
        <div className="card-header">
          <div>
            <h3 className="card-title">Patient Information</h3>
            <p className="card-subtitle">Enter patient details for the report</p>
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '14px', fontWeight: '500' }}>
              <User size={16} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
              Patient Name
            </label>
            <input type="text" className="form-input"
              placeholder="Enter patient name" value={patientName}
              onChange={(e) => setPatientName(e.target.value)} />
          </div>
          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '14px', fontWeight: '500' }}>
              <Calendar size={16} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
              Age
            </label>
            <input type="number" className="form-input"
              placeholder="Enter age" value={patientAge}
              onChange={(e) => setPatientAge(e.target.value)}
              min="0" max="150" />
          </div>
        </div>
      </div>

      {/* ══════════════════════════════════════════
          EDF + QRS MODE  (existing — unchanged)
          ══════════════════════════════════════════ */}
      {analysisMode === 'edf' && (
        <>
          <div className="grid-2">
            {/* Upload Area */}
            <div className="card">
              <div className="card-header">
                <div>
                  <h3 className="card-title">File Upload</h3>
                  <p className="card-subtitle">Drag and drop or click to browse</p>
                </div>
              </div>
              <div
                className={`upload-area ${isDragging ? 'dragging' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-input').click()}
              >
                <input type="file" id="file-input" multiple accept=".edf,.qrs"
                  style={{ display: 'none' }} onChange={handleFileSelect} />
                <UploadIcon className="upload-icon" />
                <p className="upload-text">
                  {isDragging ? 'Drop files here' : 'Drag & drop EDF or QRS files here'}
                </p>
                <p className="upload-hint">or click to browse</p>
              </div>
              <div style={{ marginTop: '16px' }}>
                <h4 style={{ fontSize: '14px', marginBottom: '8px' }}>Supported Formats:</h4>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  <span className="result-badge badge-normal">.edf</span>
                  <span className="result-badge badge-warning">.qrs</span>
                </div>
              </div>
            </div>

            {/* Selected Files */}
            <div className="card">
              <div className="card-header">
                <div>
                  <h3 className="card-title">Selected Files</h3>
                  <p className="card-subtitle">{files.length} file(s) ready for analysis</p>
                </div>
              </div>
              {files.length > 0 ? (
                <div>
                  {files.map((file, index) => (
                    <div key={index} className="result-card">
                      <FileAudio size={24} style={{ color: 'var(--primary-color)' }} />
                      <div className="result-info">
                        <div className="result-title">{file.name}</div>
                        <div className="result-subtitle">{(file.size / 1024).toFixed(2)} KB</div>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        {uploadProgress[file.name] === 'completed' && (
                          <CheckCircle size={20} style={{ color: 'var(--success-color)' }} />
                        )}
                        {uploadProgress[file.name] === 'error' && (
                          <span style={{ color: 'var(--danger-color)', fontSize: '12px' }}>Error</span>
                        )}
                        {uploadProgress[file.name] === 'uploading' && (
                          <span style={{ fontSize: '12px', color: 'var(--primary-color)' }}>Uploading...</span>
                        )}
                        <button className="btn btn-sm btn-secondary"
                          onClick={() => removeFile(index)} disabled={isUploading}>
                          Remove
                        </button>
                      </div>
                    </div>
                  ))}
                  <button className="btn btn-primary btn-lg"
                    style={{ width: '100%', marginTop: '16px' }}
                    onClick={handleUpload} disabled={isUploading}>
                    {isUploading ? 'Analyzing...' : 'Start Analysis'}
                  </button>
                </div>
              ) : (
                <div className="empty-state">
                  <FileAudio />
                  <h3>No files selected</h3>
                  <p>Upload EDF files to begin analysis</p>
                </div>
              )}
            </div>
          </div>

          {/* How It Works */}
          <div className="card" style={{ marginTop: '24px' }}>
            <div className="card-header">
              <div><h3 className="card-title">How It Works</h3></div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '24px' }}>
              {[
                { step: 1, title: 'Enter Patient Info', desc: 'Provide patient name and age for the report' },
                { step: 2, title: 'Upload ECG', desc: 'Upload EDF format ECG files along with QRS annotations' },
                { step: 3, title: 'AI Analysis', desc: 'Our DSNN model analyzes the ECG for arrhythmia patterns' },
                { step: 4, title: 'View Results', desc: 'Get detailed results with confidence scores and visualizations' },
              ].map(({ step, title, desc }) => (
                <div key={step} style={{ textAlign: 'center', padding: '16px' }}>
                  <div style={{
                    width: '48px', height: '48px', borderRadius: '50%',
                    background: 'var(--primary-color)', color: 'white',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    margin: '0 auto 12px', fontWeight: 'bold'
                  }}>{step}</div>
                  <h4>{title}</h4>
                  <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>{desc}</p>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {/* ══════════════════════════════════════════
          CSV MODE  (new)
          ══════════════════════════════════════════ */}
      {analysisMode === 'csv' && (
        <>
          <div className="grid-2">
            {/* CSV Drop Zone */}
            <div className="card">
              <div className="card-header">
                <div>
                  <h3 className="card-title">CSV File Upload</h3>
                  <p className="card-subtitle">Upload a CSV with pre-extracted ECG features</p>
                </div>
              </div>

              <div
                className={`upload-area ${csvDragOver ? 'dragging' : ''} ${csvFile ? 'has-file' : ''}`}
                onDragOver={handleCsvDragOver}
                onDragLeave={handleCsvDragLeave}
                onDrop={handleCsvDrop}
                onClick={() => document.getElementById('csv-file-input').click()}
                style={{
                  border: csvFile ? '2px dashed var(--primary-color)' : undefined,
                  background: csvFile ? 'rgba(99,102,241,0.05)' : undefined,
                }}
              >
                <input type="file" id="csv-file-input" accept=".csv"
                  style={{ display: 'none' }} onChange={handleCsvFileSelect} />
                <FileSpreadsheet className="upload-icon"
                  style={{ color: csvFile ? 'var(--primary-color)' : undefined }} />
                <p className="upload-text">
                  {csvDragOver ? 'Drop CSV here'
                    : csvFile ? csvFile.name
                    : 'Drag & drop your CSV file here'}
                </p>
                <p className="upload-hint">
                  {csvFile
                    ? `${(csvFile.size / 1024).toFixed(1)} KB — click to change`
                    : 'or click to browse — .csv only'}
                </p>
              </div>

              <div style={{ marginTop: '16px' }}>
                <h4 style={{ fontSize: '14px', marginBottom: '8px' }}>Supported Format:</h4>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  <span className="result-badge badge-normal">.csv</span>
                </div>
              </div>
            </div>

            {/* CSV Info Panel */}
            <div className="card">
              <div className="card-header">
                <div>
                  <h3 className="card-title">Analysis Details</h3>
                  <p className="card-subtitle">How CSV batch analysis works</p>
                </div>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                {[
                  { icon: '📋', title: 'Multi-row batch', desc: 'Each row = one heartbeat. All rows are analyzed together.' },
                  { icon: '🗳️', title: 'Majority voting', desc: 'The most frequent predicted class becomes the primary diagnosis.' },
                  { icon: '📊', title: 'Confidence score', desc: 'Confidence = majority class count ÷ total beats × 100.' },
                  { icon: '🔧', title: 'Required columns', desc: 'Numerical feature columns only. record/type columns are auto-dropped.' },
                ].map(({ icon, title, desc }) => (
                  <div key={title} style={{
                    display: 'flex', gap: '12px', alignItems: 'flex-start',
                    padding: '12px', borderRadius: '8px',
                    background: 'var(--background-secondary)',
                  }}>
                    <span style={{ fontSize: '20px', flexShrink: 0 }}>{icon}</span>
                    <div>
                      <div style={{ fontWeight: '600', fontSize: '13px', marginBottom: '2px' }}>{title}</div>
                      <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{desc}</div>
                    </div>
                  </div>
                ))}

                <div style={{
                  padding: '12px', borderRadius: '8px',
                  background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.3)',
                  fontSize: '12px', color: 'var(--text-secondary)',
                }}>
                  ⚠️ <strong>Note:</strong> The CSV MLP model must be trained first via <em>Model Training → CSV Dataset</em>.
                </div>

                {csvFile && (
                  <button className="btn btn-primary btn-lg" style={{ width: '100%', marginTop: '4px' }}
                    onClick={handleCsvAnalyze} disabled={isCsvAnalyzing}>
                    {isCsvAnalyzing
                      ? <><span style={{ marginRight: '8px' }}>⏳</span>Analyzing CSV...</>
                      : <><FileSpreadsheet size={18} style={{ marginRight: '8px' }} />Start CSV Analysis</>
                    }
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* CSV How It Works */}
          <div className="card" style={{ marginTop: '24px' }}>
            <div className="card-header">
              <div><h3 className="card-title">How CSV Analysis Works</h3></div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '24px' }}>
              {[
                { step: 1, title: 'Enter Patient Info', desc: 'Provide patient name and age for the report' },
                { step: 2, title: 'Upload CSV', desc: 'Upload a .csv file with pre-extracted ECG morphological features' },
                { step: 3, title: 'MLP Inference', desc: 'Our trained MLP model classifies each heartbeat row in batch' },
                { step: 4, title: 'Majority Voting', desc: 'Results are aggregated via majority vote into a final diagnosis' },
              ].map(({ step, title, desc }) => (
                <div key={step} style={{ textAlign: 'center', padding: '16px' }}>
                  <div style={{
                    width: '48px', height: '48px', borderRadius: '50%',
                    background: 'var(--primary-color)', color: 'white',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    margin: '0 auto 12px', fontWeight: 'bold'
                  }}>{step}</div>
                  <h4>{title}</h4>
                  <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>{desc}</p>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default Upload;
