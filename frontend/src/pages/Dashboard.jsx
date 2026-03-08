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
import { getDashboardStats } from '../services/api';

const Dashboard = () => {
  const [stats, setStats] = useState({
    totalTests: 0,
    normalResults: 0,
    abnormalResults: 0,
    avgConfidence: 0
  });
  const [recentResults, setRecentResults] = useState([]);
  const [classDistribution, setClassDistribution] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const data = await getDashboardStats();
      setStats(data.stats || {
        totalTests: 0,
        normalResults: 0,
        abnormalResults: 0,
        avgConfidence: 0
      });
      setRecentResults(data.recent_results || []);
      setClassDistribution(data.class_distribution || []);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setStats({
        totalTests: 0,
        normalResults: 0,
        abnormalResults: 0,
        avgConfidence: 0
      });
      setRecentResults([]);
      setClassDistribution([]);
    } finally {
      setLoading(false);
    }
  };

  const COLORS = ['#10b981', '#ef4444', '#f59e0b', '#8b5cf6', '#ec4899', '#6366f1'];

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

      {loading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <p>Loading dashboard...</p>
        </div>
      )}

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
              <span>Total processed</span>
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
              <span>{((stats.normalResults / (stats.totalTests || 1)) * 100 || 0).toFixed(1)}% of total</span>
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
              <span>{((stats.abnormalResults / (stats.totalTests || 1)) * 100 || 0).toFixed(1)}% of total</span>
            </div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon blue">
            <Activity size={24} />
          </div>
          <div className="stat-content">
            <div className="stat-label">Avg. Confidence</div>
            <div className="stat-value">{stats.avgConfidence.toFixed(1)}%</div>
            <div className="stat-change positive">
              <TrendingUp size={14} />
              <span>Model confidence</span>
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
          
          {classDistribution.length > 0 ? (
            <>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height={200}>
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
            </>
          ) : (
            <div className="empty-state">
              <FileText />
              <h3>No data yet</h3>
              <p>Run analysis to see class distribution</p>
            </div>
          )}
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <h3 className="card-title">Analysis Trend</h3>
              <p className="card-subtitle">Normal vs Abnormal results</p>
            </div>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={recentResults.length > 0 ? [
                { month: 'Current', normal: stats.normalResults, abnormal: stats.abnormalResults }
              ] : [
                { month: 'No Data', normal: 0, abnormal: 0 }
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
                      <span className={`result-badge ${result.is_normal ? 'badge-normal' : 'badge-warning'}`}>
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
