import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload as UploadIcon, FileAudio, AlertCircle, CheckCircle } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';

const Upload = () => {
  const navigate = useNavigate();
  const [isDragging, setIsDragging] = useState(false);
  const [files, setFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({});

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

    setIsUploading(true);
    const uploadedIds = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      setUploadProgress(prev => ({ ...prev, [file.name]: 'uploading' }));

      try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('patient_id', `patient_${Date.now()}_${i}`);

        const response = await axios.post('http://localhost:5000/api/analyze', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(prev => ({ ...prev, [file.name]: percentCompleted }));
          }
        });

        uploadedIds.push(response.data.id);
        setUploadProgress(prev => ({ ...prev, [file.name]: 'completed' }));
        toast.success(`Analysis complete for ${file.name}`);
      } catch (error) {
        setUploadProgress(prev => ({ ...prev, [file.name]: 'error' }));
        toast.error(`Failed to analyze ${file.name}`);
      }
    }

    setIsUploading(false);
    
    if (uploadedIds.length > 0) {
      navigate(`/results/${uploadedIds[0]}`);
    }
  };

  return (
    <div className="upload-page">
      <div className="page-header">
        <div>
          <h1>Upload ECG Data</h1>
          <p className="text-secondary">Upload EDF files for arrhythmia detection</p>
        </div>
      </div>

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
            <input
              type="file"
              id="file-input"
              multiple
              accept=".edf,.qrs"
              style={{ display: 'none' }}
              onChange={handleFileSelect}
            />
            <UploadIcon className="upload-icon" />
            <p className="upload-text">
              {isDragging ? 'Drop files here' : 'Drag & drop EDF or QRS files here'}
            </p>
            <p className="upload-hint">or click to browse</p>
          </div>

          {/* Supported Formats */}
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
                    <div className="result-subtitle">
                      {(file.size / 1024).toFixed(2)} KB
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {uploadProgress[file.name] === 'completed' && (
                      <CheckCircle size={20} style={{ color: 'var(--success-color)' }} />
                    )}
                    {uploadProgress[file.name] === 'error' && (
                      <AlertCircle size={20} style={{ color: 'var(--danger-color)' }} />
                    )}
                    {uploadProgress[file.name] === 'uploading' && (
                      <span style={{ fontSize: '12px', color: 'var(--primary-color)' }}>
                        Uploading...
                      </span>
                    )}
                    <button
                      className="btn btn-sm btn-secondary"
                      onClick={() => removeFile(index)}
                      disabled={isUploading}
                    >
                      Remove
                    </button>
                  </div>
                </div>
              ))}

              <button
                className="btn btn-primary btn-lg"
                style={{ width: '100%', marginTop: '16px' }}
                onClick={handleUpload}
                disabled={isUploading}
              >
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

      {/* Instructions */}
      <div className="card" style={{ marginTop: '24px' }}>
        <div className="card-header">
          <div>
            <h3 className="card-title">How It Works</h3>
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '24px' }}>
          <div style={{ textAlign: 'center', padding: '16px' }}>
            <div style={{ 
              width: '48px', 
              height: '48px', 
              borderRadius: '50%', 
              background: 'var(--primary-color)',
              color: 'white',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 12px',
              fontWeight: 'bold'
            }}>1</div>
            <h4>Upload ECG</h4>
            <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
              Upload EDF format ECG files along with QRS annotations
            </p>
          </div>
          <div style={{ textAlign: 'center', padding: '16px' }}>
            <div style={{ 
              width: '48px', 
              height: '48px', 
              borderRadius: '50%', 
              background: 'var(--primary-color)',
              color: 'white',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 12px',
              fontWeight: 'bold'
            }}>2</div>
            <h4>AI Analysis</h4>
            <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
              Our DSNN model analyzes the ECG for arrhythmia patterns
            </p>
          </div>
          <div style={{ textAlign: 'center', padding: '16px' }}>
            <div style={{ 
              width: '48px', 
              height: '48px', 
              borderRadius: '50%', 
              background: 'var(--primary-color)',
              color: 'white',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 12px',
              fontWeight: 'bold'
            }}>3</div>
            <h4>View Results</h4>
            <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
              Get detailed results with confidence scores and visualizations
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Upload;

