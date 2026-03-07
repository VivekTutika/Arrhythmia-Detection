import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { 
  ArrowLeft, 
  Download, 
  Share2, 
  Activity, 
  Heart,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  FileText
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import axios from 'axios';

const Results = () => {
  const { id } = useParams();
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchResult();
  }, [id]);

  const fetchResult = async () => {
    try {
      const response = await axios.get(`http://localhost:5000/api/results/${id}`);
      setResult(response.data);
    } catch {
      // Mock data for demo
      setResult({
        id: id,
        file_name: 'patient_sample.edf',
        status: 'completed',
        created_at: '2024-01-15T10:30:00Z',
        result: {
          primary_diagnosis: 'Normal Sinus Rhythm',
          confidence: 92.5,
          is_normal: true,
          segments_analyzed: 156,
          predictions: {
            'Normal Sinus Rhythm': 85,
            'Atrial Fibrillation': 5,
            'Ventricular Arrhythmia': 3,
            'Conduction Block': 2,
            'Premature Contraction': 3,
            'ST Segment Abnormality': 2
          }
        },
        ecg_metrics: {
          heart_rate: 72,
          rr_interval: 833,
          hrv: 45,
          p_wave: 0.12,
          qrs_complex: 0.08,
          qt_interval: 0.38
        },
        recommendations: [
          'Continue regular cardiac checkups',
          'Maintain healthy lifestyle',
          'No immediate medical intervention required'
        ]
      });
    } finally {
      setLoading(false);
    }
  };

  const getResultBadge = () => {
    if (!result) return null;
    if (result.result.is_normal) {
      return <span className="result-badge badge-normal"><CheckCircle size={14} /> Normal</span>;
    }
    return <span className="result-badge badge-danger"><AlertTriangle size={14} /> Abnormal</span>;
  };

  const pieData = result ? Object.entries(result.result.predictions).map(([name, value]) => ({
    name,
    value
  })) : [];

  const COLORS = ['#10b981', '#ef4444', '#f59e0b', '#8b5cf6', '#ec4899', '#64748b'];

  if (loading) {
    return (
      <div className="results-page">
        <div className="card" style={{ textAlign: 'center', padding: '48px' }}>
          <Activity size={48} style={{ color: 'var(--primary-color)', marginBottom: '16px' }} />
          <p>Loading results...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="results-page">
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <Link to="/history" className="btn btn-secondary">
            <ArrowLeft size={18} />
            Back
          </Link>
          <div>
            <h1>Analysis Results</h1>
            <p className="text-secondary">{result.file_name}</p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button className="btn btn-secondary">
            <Download size={18} />
            Export PDF
          </button>
          <button className="btn btn-secondary">
            <Share2 size={18} />
            Share
          </button>
        </div>
      </div>

      {/* Main Result Card */}
      <div className="card" style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <div style={{ 
              width: '64px', 
              height: '64px', 
              borderRadius: '50%', 
              background: result.result.is_normal ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              {result.result.is_normal ? (
                <CheckCircle size={32} style={{ color: 'var(--success-color)' }} />
              ) : (
                <XCircle size={32} style={{ color: 'var(--danger-color)' }} />
              )}
            </div>
            <div>
              <h2 style={{ marginBottom: '4px' }}>{result.result.primary_diagnosis}</h2>
              {getResultBadge()}
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '36px', fontWeight: 'bold', color: 'var(--primary-color)' }}>
              {result.result.confidence}%
            </div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>Confidence Score</div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '16px' }}>
          <div className="stat-card">
            <div className="stat-icon blue">
              <Activity size={20} />
            </div>
            <div className="stat-content">
              <div className="stat-label">Heart Rate</div>
              <div className="stat-value">{result.ecg_metrics.heart_rate} <span style={{ fontSize: '16px' }}>BPM</span></div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon green">
              <Heart size={20} />
            </div>
            <div className="stat-content">
              <div className="stat-label">RR Interval</div>
              <div className="stat-value">{result.ecg_metrics.rr_interval} <span style={{ fontSize: '16px' }}>ms</span></div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon orange">
              <Activity size={20} />
            </div>
            <div className="stat-content">
              <div className="stat-label">HRV</div>
              <div className="stat-value">{result.ecg_metrics.hrv} <span style={{ fontSize: '16px' }}>ms</span></div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon blue">
              <Clock size={20} />
            </div>
            <div className="stat-content">
              <div className="stat-label">Segments</div>
              <div className="stat-value">{result.result.segments_analyzed}</div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid-2" style={{ marginBottom: '24px' }}>
        {/* Class Distribution */}
        <div className="card">
          <div className="card-header">
            <div>
              <h3 className="card-title">Prediction Distribution</h3>
              <p className="card-subtitle">Probability across all classes</p>
            </div>
          </div>
          <div style={{ height: '250px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={90}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div style={{ marginTop: '16px' }}>
            {pieData.map((item, index) => (
              <div key={index} style={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'space-between',
                padding: '8px 0',
                borderBottom: '1px solid var(--border-color)'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: COLORS[index] }} />
                  <span style={{ fontSize: '14px' }}>{item.name}</span>
                </div>
                <span style={{ fontWeight: '600' }}>{item.value}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* ECG Waveform Metrics */}
        <div className="card">
          <div className="card-header">
            <div>
              <h3 className="card-title">ECG Waveform Analysis</h3>
              <p className="card-subtitle">Detailed wave measurements</p>
            </div>
          </div>
          <div style={{ display: 'grid', gap: '16px' }}>
            {[
              { label: 'P Wave Duration', value: result.ecg_metrics.p_wave, unit: 's' },
              { label: 'QRS Complex', value: result.ecg_metrics.qrs_complex, unit: 's' },
              { label: 'QT Interval', value: result.ecg_metrics.qt_interval, unit: 's' },
              { label: 'RR Interval', value: result.ecg_metrics.rr_interval / 1000, unit: 's' },
            ].map((metric, index) => (
              <div key={index} style={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'space-between',
                padding: '12px',
                background: 'var(--background-secondary)',
                borderRadius: '8px'
              }}>
                <span>{metric.label}</span>
                <span style={{ fontWeight: '600', color: 'var(--primary-color)' }}>
                  {metric.value.toFixed(3)} {metric.unit}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recommendations */}
      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title">Recommendations</h3>
            <p className="card-subtitle">Based on the analysis results</p>
          </div>
        </div>
        <div style={{ display: 'grid', gap: '12px' }}>
          {result.recommendations.map((rec, index) => (
            <div key={index} style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '12px',
              padding: '12px',
              background: 'var(--background-secondary)',
              borderRadius: '8px',
              borderLeft: '3px solid var(--primary-color)'
            }}>
              <FileText size={18} style={{ color: 'var(--primary-color)' }} />
              <span>{rec}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Results;

