import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { FileText, Search, Filter, Download, Trash2, Eye, ChevronLeft, ChevronRight } from 'lucide-react';
import axios from 'axios';

const History = () => {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  useEffect(() => {
    fetchResults();
  }, [currentPage, filterStatus]);

  const fetchResults = async () => {
    setLoading(true);
    try {
      const response = await axios.get('http://localhost:5000/api/results', {
        params: { page: currentPage, status: filterStatus }
      });
      setResults(response.data.results);
      setTotalPages(response.data.total_pages);
    } catch {
      // Mock data
      setResults([
        { id: 1, file_name: 'patient_001.edf', result: 'Normal Sinus Rhythm', confidence: 92.5, is_normal: true, created_at: '2024-01-15T10:30:00Z' },
        { id: 2, file_name: 'patient_002.edf', result: 'Atrial Fibrillation', confidence: 88.3, is_normal: false, created_at: '2024-01-14T14:20:00Z' },
        { id: 3, file_name: 'patient_003.edf', result: 'Normal Sinus Rhythm', confidence: 95.1, is_normal: true, created_at: '2024-01-14T09:15:00Z' },
        { id: 4, file_name: 'patient_004.edf', result: 'Ventricular Arrhythmia', confidence: 78.9, is_normal: false, created_at: '2024-01-13T16:45:00Z' },
        { id: 5, file_name: 'patient_005.edf', result: 'Conduction Block', confidence: 84.2, is_normal: false, created_at: '2024-01-12T11:00:00Z' },
      ]);
      setTotalPages(3);
    } finally {
      setLoading(false);
    }
  };

  const filteredResults = results.filter(r => 
    r.file_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    r.result.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const deleteResult = async (id) => {
    if (!confirm('Are you sure you want to delete this result?')) return;
    
    try {
      await axios.delete(`http://localhost:5000/api/results/${id}`);
      setResults(results.filter(r => r.id !== id));
    } catch {
      setResults(results.filter(r => r.id !== id));
    }
  };

  return (
    <div className="history-page">
      <div className="page-header">
        <div>
          <h1>Analysis History</h1>
          <p className="text-secondary">View all previous ECG analyses</p>
        </div>
      </div>

      {/* Filters */}
      <div className="card" style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', alignItems: 'center' }}>
          <div style={{ flex: 1, minWidth: '200px', position: 'relative' }}>
            <Search size={18} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-secondary)' }} />
            <input
              type="text"
              className="form-input"
              placeholder="Search by file name or result..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{ paddingLeft: '40px' }}
            />
          </div>
          
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              className={`btn ${filterStatus === 'all' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setFilterStatus('all')}
            >
              All
            </button>
            <button
              className={`btn ${filterStatus === 'normal' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setFilterStatus('normal')}
            >
              Normal
            </button>
            <button
              className={`btn ${filterStatus === 'abnormal' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setFilterStatus('abnormal')}
            >
              Abnormal
            </button>
          </div>
        </div>
      </div>

      {/* Results Table */}
      <div className="card">
        {loading ? (
          <div style={{ textAlign: 'center', padding: '48px' }}>
            <p>Loading results...</p>
          </div>
        ) : filteredResults.length > 0 ? (
          <>
            <div className="table-container">
              <table className="table">
                <thead>
                  <tr>
                    <th>File Name</th>
                    <th>Result</th>
                    <th>Confidence</th>
                    <th>Date</th>
                    <th>Status</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredResults.map((result) => (
                    <tr key={result.id}>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <FileText size={16} style={{ color: 'var(--primary-color)' }} />
                          <span style={{ fontWeight: '500' }}>{result.file_name}</span>
                        </div>
                      </td>
                      <td>{result.result}</td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <div className="progress-bar" style={{ width: '60px' }}>
                            <div className="progress-fill" style={{ width: `${result.confidence}%` }} />
                          </div>
                          <span>{result.confidence}%</span>
                        </div>
                      </td>
                      <td>{formatDate(result.created_at)}</td>
                      <td>
                        <span className={`result-badge ${result.is_normal ? 'badge-normal' : 'badge-warning'}`}>
                          {result.is_normal ? 'Normal' : 'Abnormal'}
                        </span>
                      </td>
                      <td>
                        <div style={{ display: 'flex', gap: '4px' }}>
                          <Link to={`/results/${result.id}`} className="btn btn-sm btn-secondary">
                            <Eye size={14} />
                          </Link>
                          <button className="btn btn-sm btn-secondary">
                            <Download size={14} />
                          </button>
                          <button 
                            className="btn btn-sm btn-secondary"
                            onClick={() => deleteResult(result.id)}
                            style={{ color: 'var(--danger-color)' }}
                          >
                            <Trash2 size={14} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                marginTop: '16px',
                paddingTop: '16px',
                borderTop: '1px solid var(--border-color)'
              }}>
                <span style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>
                  Page {currentPage} of {totalPages}
                </span>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                    disabled={currentPage === 1}
                  >
                    <ChevronLeft size={16} />
                    Previous
                  </button>
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                    disabled={currentPage === totalPages}
                  >
                    Next
                    <ChevronRight size={16} />
                  </button>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="empty-state">
            <FileText />
            <h3>No results found</h3>
            <p>Try adjusting your search or filters</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default History;

