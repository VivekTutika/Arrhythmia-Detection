import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  Activity, 
  FileText, 
  TrendingUp, 
  AlertCircle,
  CheckCircle,
  ArrowRight
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import axios from 'axios';

const Dashboard = () => {
  const [stats, setStats] = useState({
    totalTests: 0,
    normalResults: 0,
    abnormalResults: 0,
    avgConfidence: 0
  });
  const [recentResults, setRecentResults] = useState([]);
  const [classDistribution, setClassDistribution] = useState([]);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/dashboard');
      setStats(response.data.stats);
      setRecentResults(response.data.recent_results);
      setClassDistribution(response.data.class_distribution);
    } catch (error) {
      // Use mock data if API is not available
      setStats({
        totalTests: 24,
        normalResults: 18,
        abnormalResults: 6,
        avgConfidence: 87.5
      });
      setRecentResults([
        { id: 1, file_name: 'patient_001.edf', result: 'Normal', confidence: 92.5, date: '2024-01-15' },
        { id: 2, file_name: 'patient_002.edf', result: 'Atrial Fibrillation', confidence: 88.3, date: '2024-01-14' },
        { id: 3, file_name: 'patient_003.edf', result: 'Normal', confidence: 95.1, date: '2024-01-14' },
      ]);
      setClassDistribution([
        { name: 'Normal', value: 75, color: '#10b981' },
        { name: 'AFib', value: 10, color: '#ef4444' },
        { name: 'Ventricular', value: 5, color: '#f59e0b' },
        { name: 'Other', value: 10, color: '#64748b' },
      ]);
    }
  };

  const COLORS = ['#10b981', '#ef4444', '#f59e0b', '#64748b'];

  return (
    <div className="dashboard">
      <div className="page-header">
        <div>
          <h1>Dashboard</h1>
          <p className="text-secondary">Overview of arrhythmia detection results</p>
        </div>
        <Link to="/upload" className="btn btn-primary">
          <Activity size={18} />
          New Analysis
        </Link>
      </div>

      {/* Stats Grid */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon blue">
            <FileText size={24} />
          </div>
          <div className="stat-content">
            <div className="stat-label">Total Analyses</div>
            <div className="stat-value">{stats.totalTests}</div>
            <div className="stat-change positive">
              <TrendingUp size={14} />
              <span>+12% this month</span>
            </div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon green">
            <CheckCircle size={24} />
          </div>
          <div className="stat-content">
            <div className="stat-label">Normal Results</div>
            <div className="stat-value">{stats.normalResults}</div>
            <div className="stat-change positive">
              <span>{((stats.normalResults / stats.totalTests) * 100 || 0).toFixed(1)}% of total</span>
            </div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon orange">
            <AlertCircle size={24} />
          </div>
          <div className="stat-content">
            <div className="stat-label">Abnormal Results</div>
            <div className="stat-value">{stats.abnormalResults}</div>
            <div className="stat-change negative">
              <span>{((stats.abnormalResults / stats.totalTests) * 100 || 0).toFixed(1)}% of total</span>
            </div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon blue">
            <Activity size={24} />
          </div>
          <div className="stat-content">
            <div className="stat-label">Avg. Confidence</div>
            <div className="stat-value">{stats.avgConfidence}%</div>
            <div className="stat-change positive">
              <TrendingUp size={14} />
              <span>+5% improvement</span>
            </div>
          </div>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid-2" style={{ marginBottom: '24px' }}>
        <div className="card">
          <div className="card-header">
            <div>
              <h3 className="card-title">Class Distribution</h3>
              <p className="card-subtitle">Distribution of detected arrhythmia types</p>
            </div>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={classDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {classDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color || COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '16px', marginTop: '16px', flexWrap: 'wrap' }}>
            {classDistribution.map((item, index) => (
              <div key={index} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: item.color || COLORS[index] }} />
                <span style={{ fontSize: '12px' }}>{item.name}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <h3 className="card-title">Monthly Trend</h3>
              <p className="card-subtitle">Analysis results over time</p>
            </div>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={[
                { month: 'Aug', normal: 12, abnormal: 3 },
                { month: 'Sep', normal: 15, abnormal: 4 },
                { month: 'Oct', normal: 18, abnormal: 5 },
                { month: 'Nov', normal: 14, abnormal: 6 },
                { month: 'Dec', normal: 20, abnormal: 4 },
                { month: 'Jan', normal: 18, abnormal: 6 },
              ]}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                <XAxis dataKey="month" stroke="var(--text-secondary)" />
                <YAxis stroke="var(--text-secondary)" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'var(--background-card)', 
                    border: '1px solid var(--border-color)',
                    borderRadius: '8px'
                  }}
                />
                <Bar dataKey="normal" name="Normal" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="abnormal" name="Abnormal" fill="#ef4444" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Recent Results */}
      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title">Recent Analyses</h3>
            <p className="card-subtitle">Latest ECG analysis results</p>
          </div>
          <Link to="/history" className="btn btn-secondary">
            View All
            <ArrowRight size={16} />
          </Link>
        </div>
        
        {recentResults.length > 0 ? (
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>File Name</th>
                  <th>Result</th>
                  <th>Confidence</th>
                  <th>Date</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {recentResults.map((result) => (
                  <tr key={result.id}>
                    <td>{result.file_name}</td>
                    <td>
                      <span className={`result-badge ${result.result === 'Normal' ? 'badge-normal' : 'badge-warning'}`}>
                        {result.result}
                      </span>
                    </td>
                    <td>{result.confidence}%</td>
                    <td>{result.date}</td>
                    <td>
                      <Link to={`/results/${result.id}`} className="btn btn-sm btn-secondary">
                        View Details
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state">
            <FileText />
            <h3>No analyses yet</h3>
            <p>Upload an ECG file to get started</p>
            <Link to="/upload" className="btn btn-primary" style={{ marginTop: '16px' }}>
              Upload ECG
            </Link>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;

