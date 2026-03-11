import { useState, useEffect } from 'react';
import { useParams, Link, useLocation } from 'react-router-dom';
import {
  ArrowLeft,
  Download,
  Activity,
  Heart,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  FileText,
  AlertCircle,
  User,
  Calendar
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { getResult } from '../services/api';
import { jsPDF } from 'jspdf';

const Results = () => {
  const { id } = useParams();
  const location = useLocation();
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    fetchResult();
  }, [id]);

  const fetchResult = async () => {
    try {
      setLoading(true);
      setError(null);

      // If result was handed through React Router state (because autoSave is disabled),
      // we already have it in memory! Read it straight from the pass-through.
      if (location.state && location.state.resultData) {
        setResult(location.state.resultData);
        setLoading(false);
        return;
      }

      const data = await getResult(id);
      setResult(data);
    } catch (err) {
      console.error('Error fetching result:', err);
      // Give a highly detailed error indicating whether it was not found
      setError('Could not load results. If "Auto-save Results" is disabled, reloading this page will lose your results. Please re-run the analysis.');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  // PDF Export Function - Full Single Page with Disclaimer at Footer
  const exportToPDF = async () => {
    if (!result) return;

    setExporting(true);

    try {
      const doc = new jsPDF();
      const pageWidth = doc.internal.pageSize.getWidth();
      const pageHeight = doc.internal.pageSize.getHeight();
      const margin = 15;
      let yPos = margin;

      // Header
      doc.setFillColor(102, 126, 234);
      doc.rect(0, 0, pageWidth, 28, 'F');

      doc.setTextColor(255, 255, 255);
      doc.setFontSize(18);
      doc.setFont('helvetica', 'bold');
      doc.text('Arrhythmia Detection Report', pageWidth / 2, 14, { align: 'center' });
      doc.setFontSize(10);
      doc.setFont('helvetica', 'normal');
      doc.text('ECG Analysis Results - Deep Spiking Neural Network (DSNN)', pageWidth / 2, 22, { align: 'center' });

      yPos = 36;

      // Patient Info Section
      doc.setTextColor(0, 0, 0);
      doc.setFontSize(12);
      doc.setFont('helvetica', 'bold');
      doc.text('PATIENT INFORMATION', margin, yPos);
      yPos += 8;

      doc.setDrawColor(200, 200, 200);
      doc.line(margin, yPos - 3, pageWidth - margin, yPos - 3);

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(11);
      const patientName = result.patient_name || 'Anonymous';
      const patientAge = result.patient_age || 'N/A';
      const fileName = result.file_name || 'Unknown';
      // Convert to IST (UTC+5:30) for display
      const createdAt = new Date(result.created_at);
      const displayOffset = 0;
      const displayDate = new Date(createdAt.getTime() + displayOffset);
      const analysisDate = displayDate.toLocaleDateString('en-US', {
        year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit', timeZone: 'Asia/Kolkata'
      }) + ' IST';

      doc.text(`Patient Name: ${patientName}`, margin, yPos + 3);
      doc.text(`Age: ${patientAge} years`, margin + 90, yPos + 3);
      yPos += 8;
      doc.text(`Analysis Date: ${analysisDate}`, margin, yPos + 1);
      doc.text(`File Name: ${fileName}`, margin + 90, yPos + 1);
      yPos += 12;

      // Diagnosis Result Section
      doc.setFontSize(12);
      doc.setFont('helvetica', 'bold');
      doc.text('DIAGNOSIS RESULT', margin, yPos);
      yPos += 8;

      doc.line(margin, yPos - 3, pageWidth - margin, yPos - 3);
      yPos += 3;

      const isNormal = result.result?.is_normal;
      if (isNormal) {
        doc.setFillColor(16, 185, 129);
      } else {
        doc.setFillColor(239, 68, 68);
      }
      doc.roundedRect(margin, yPos, pageWidth - 2 * margin, 24, 2, 2, 'F');

      doc.setTextColor(255, 255, 255);
      doc.setFontSize(15);
      doc.setFont('helvetica', 'bold');
      doc.text(result.result?.primary_diagnosis || 'Unknown', margin + 5, yPos + 10);
      doc.setFontSize(11);
      doc.text(`Confidence: ${result.result?.confidence || 0}%`, margin + 5, yPos + 19);

      doc.setFont('helvetica', 'normal');
      const segmentsText = `Segments Analyzed: ${result.result?.segments_analyzed || 0}`;
      doc.text(segmentsText, pageWidth - margin - doc.getTextWidth(segmentsText) - 5, yPos + 19);

      yPos += 32;

      // ECG Metrics Section - Full Details
      doc.setTextColor(0, 0, 0);
      doc.setFontSize(12);
      doc.setFont('helvetica', 'bold');
      doc.text('ECG METRICS & MEASUREMENTS', margin, yPos);
      yPos += 8;

      doc.line(margin, yPos - 3, pageWidth - margin, yPos - 3);
      yPos += 5;

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(10);

      // Create a table-like structure for metrics with even spacing
      const metricsData = [
        ['Heart Rate', `${result.ecg_metrics?.heart_rate || 'N/A'} BPM`, 'RR Interval', `${result.ecg_metrics?.rr_interval || 'N/A'} ms`],
        ['HRV (Heart Rate Variability)', `${result.ecg_metrics?.hrv || 'N/A'} ms`, 'P Wave Duration', `${result.ecg_metrics?.p_wave?.toFixed(3) || 'N/A'} s`],
        ['QRS Complex', `${result.ecg_metrics?.qrs_complex?.toFixed(3) || 'N/A'} s`, 'QT Interval', `${result.ecg_metrics?.qt_interval?.toFixed(3) || 'N/A'} s`],
        ['R-Peaks Found', `${result.ecg_metrics?.r_peaks || 0}`, 'Segments Assessed', `${result.result?.segments_analyzed || 0}`]
      ];

      metricsData.forEach(row => {
        doc.text(`${row[0]}:`, margin, yPos);
        doc.setFont('helvetica', 'bold');
        doc.text(row[1], margin + 55, yPos);
        doc.setFont('helvetica', 'normal');
        doc.text(`${row[2]}:`, margin + 100, yPos);
        doc.setFont('helvetica', 'bold');
        doc.text(row[3], margin + 140, yPos);
        doc.setFont('helvetica', 'normal');
        yPos += 8;
      });

      yPos += 8;

      // Heart Rate Insights Section
      if (result.ecg_metrics?.hr_categories && result.ecg_metrics.hr_categories.length > 0) {
        doc.setFontSize(12);
        doc.setFont('helvetica', 'bold');
        doc.text('HEART RATE INSIGHTS', margin, yPos);
        yPos += 8;

        doc.line(margin, yPos - 3, pageWidth - margin, yPos - 3);
        yPos += 5;

        doc.setFontSize(10);
        doc.setFont('helvetica', 'normal');

        result.ecg_metrics.hr_categories.forEach((category) => {
          doc.text(`> ${category}`, margin, yPos);
          yPos += 7;
        });

        yPos += 8;
      }

      // Recommendations Section
      doc.setFontSize(12);
      doc.setFont('helvetica', 'bold');
      doc.text('RECOMMENDATIONS', margin, yPos);
      yPos += 8;

      doc.line(margin, yPos - 3, pageWidth - margin, yPos - 3);
      yPos += 5;

      doc.setFontSize(10);
      doc.setFont('helvetica', 'normal');

      (result.recommendations || []).forEach((rec, index) => {
        doc.text(`${index + 1}. ${rec}`, margin, yPos);
        yPos += 7;
      });

      // Calculate footer position - place disclaimer at bottom
      const footerY = pageHeight - 30;

      // Disclaimer Box at Footer
      doc.setFillColor(255, 250, 240);
      doc.setDrawColor(255, 200, 100);
      doc.roundedRect(margin, footerY, pageWidth - 2 * margin, 20, 2, 2, 'FD');

      doc.setTextColor(180, 100, 0);
      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      doc.text('WARNING', margin + 3, footerY + 6);

      doc.setFont('helvetica', 'italic');
      doc.setFontSize(9);
      doc.setTextColor(80, 80, 80);
      doc.text('This is a study-purpose detection system powered by Deep Spiking Neural Network (DSNN).', margin + 3, footerY + 12);
      doc.text('Don\'t solely rely on this analysis for any medical decisions. Consult a qualified healthcare professional.', margin + 3, footerY + 18);

      // Save with new naming convention
      const now = new Date();
      const istOffset = 5.5 * 60 * 60 * 1000;
      const istDate = new Date(now.getTime() + istOffset);
      const timestamp = istDate.toISOString().replace(/[:.]/g, '-').slice(0, 19);
      const cleanFileName = fileName.replace(/\.[^/.]+$/, '');
      const newFileName = `AD_Report_${patientName.replace(/\s+/g, '_')}_${patientAge}_${cleanFileName}_${timestamp}.pdf`;
      doc.save(newFileName);

    } catch (err) {
      console.error('Error generating PDF:', err);
      alert('Failed to generate PDF. Please try again.');
    } finally {
      setExporting(false);
    }
  };

  const getResultBadge = () => {
    if (!result || !result.result) return null;
    if (result.result.is_normal) {
      return <span className="result-badge badge-normal"><CheckCircle size={14} /> Normal</span>;
    }
    return <span className="result-badge badge-danger"><AlertTriangle size={14} /> Abnormal</span>;
  };

  const pieData = result && result.result && result.result.predictions
    ? Object.entries(result.result.predictions).map(([name, value]) => ({
      name,
      value
    }))
    : [];

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

  if (error) {
    return (
      <div className="results-page">
        <div className="page-header">
          <Link to="/history" className="btn btn-secondary">
            <ArrowLeft size={18} />
            Back to History
          </Link>
        </div>
        <div className="card" style={{ textAlign: 'center', padding: '48px' }}>
          <AlertCircle size={48} style={{ color: 'var(--danger-color)', marginBottom: '16px' }} />
          <h3>Error Loading Results</h3>
          <p className="text-secondary">{error}</p>
          <Link to="/upload" className="btn btn-primary" style={{ marginTop: '16px' }}>
            Upload New ECG
          </Link>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="results-page">
        <div className="page-header">
          <Link to="/history" className="btn btn-secondary">
            <ArrowLeft size={18} />
            Back to History
          </Link>
        </div>
        <div className="card" style={{ textAlign: 'center', padding: '48px' }}>
          <FileText size={48} style={{ color: 'var(--text-secondary)', marginBottom: '16px' }} />
          <h3>No Results Found</h3>
          <p className="text-secondary">The requested analysis could not be found.</p>
          <Link to="/upload" className="btn btn-primary" style={{ marginTop: '16px' }}>
            Upload New ECG
          </Link>
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
          <button
            className="btn btn-secondary"
            onClick={exportToPDF}
            disabled={exporting}
          >
            <Download size={18} />
            {exporting ? 'Generating...' : 'Export PDF'}
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
              background: result.result?.is_normal ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              {result.result?.is_normal ? (
                <CheckCircle size={32} style={{ color: 'var(--success-color)' }} />
              ) : (
                <XCircle size={32} style={{ color: 'var(--danger-color)' }} />
              )}
            </div>
            <div>
              <h2 style={{ marginBottom: '4px' }}>{result.result?.primary_diagnosis || 'Unknown'}</h2>
              {getResultBadge()}
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '36px', fontWeight: 'bold', color: 'var(--primary-color)' }}>
              {result.result?.confidence || 0}%
            </div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>Confidence Score</div>
          </div>
        </div>

        {/* Patient Info Banner */}
        <div style={{
          display: 'flex',
          gap: '24px',
          padding: '12px 16px',
          background: 'var(--background-secondary)',
          borderRadius: '8px',
          marginBottom: '16px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <User size={16} style={{ color: 'var(--primary-color)' }} />
            <span style={{ fontWeight: '500' }}>{result.patient_name || 'Anonymous'}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Calendar size={16} style={{ color: 'var(--primary-color)' }} />
            <span style={{ fontWeight: '500' }}>Age: {result.patient_age || 'N/A'}</span>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '16px' }}>
          <div className="stat-card">
            <div className="stat-icon blue">
              <Activity size={20} />
            </div>
            <div className="stat-content">
              <div className="stat-label">Heart Rate</div>
              <div className="stat-value">{result.ecg_metrics?.heart_rate || 'N/A'} <span style={{ fontSize: '16px' }}>BPM</span></div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon green">
              <Heart size={20} />
            </div>
            <div className="stat-content">
              <div className="stat-label">RR Interval</div>
              <div className="stat-value">{result.ecg_metrics?.rr_interval || 'N/A'} <span style={{ fontSize: '16px' }}>ms</span></div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon orange">
              <Activity size={20} />
            </div>
            <div className="stat-content">
              <div className="stat-label">HRV</div>
              <div className="stat-value">{result.ecg_metrics?.hrv || 'N/A'} <span style={{ fontSize: '16px' }}>ms</span></div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon red">
              <Activity size={20} />
            </div>
            <div className="stat-content">
              <div className="stat-label">R-Peaks</div>
              <div className="stat-value">{result.ecg_metrics?.r_peaks || 0}</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon blue">
              <Clock size={20} />
            </div>
            <div className="stat-content">
              <div className="stat-label">Segments</div>
              <div className="stat-value">{result.result?.segments_analyzed || 0}</div>
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
          {pieData.length > 0 ? (
            <>
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
            </>
          ) : (
            <div className="empty-state">
              <p>No prediction data available</p>
            </div>
          )}
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
              { label: 'P Wave Duration', value: result.ecg_metrics?.p_wave, unit: 's' },
              { label: 'QRS Complex', value: result.ecg_metrics?.qrs_complex, unit: 's' },
              { label: 'QT Interval', value: result.ecg_metrics?.qt_interval, unit: 's' },
              { label: 'RR Interval', value: result.ecg_metrics?.rr_interval ? result.ecg_metrics.rr_interval / 1000 : null, unit: 's' },
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
                  {metric.value !== null && metric.value !== undefined ? `${metric.value.toFixed(3)} ${metric.unit}` : 'N/A'}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Heart Rate Category Insights */}
      {result.ecg_metrics?.hr_categories && result.ecg_metrics.hr_categories.length > 0 && (
        <div className="card" style={{ marginBottom: '24px' }}>
          <div className="card-header">
            <div>
              <h3 className="card-title">Heart Rate Insights</h3>
              <p className="card-subtitle">Possible categories based on heart rate calculations</p>
            </div>
          </div>
          <div style={{ display: 'grid', gap: '12px' }}>
            {result.ecg_metrics.hr_categories.map((category, index) => (
              <div key={index} style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '12px',
                background: 'var(--background-secondary)',
                borderRadius: '8px',
                borderLeft: '3px solid var(--warning-color, #f59e0b)'
              }}>
                <Heart size={18} style={{ color: 'var(--warning-color, #f59e0b)' }} />
                <span>{category}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title">Recommendations</h3>
            <p className="card-subtitle">Based on the analysis results</p>
          </div>
        </div>
        <div style={{ display: 'grid', gap: '12px' }}>
          {(result.recommendations || []).map((rec, index) => (
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
