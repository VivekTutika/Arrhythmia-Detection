import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { FileText, Search, Download, Trash2, Eye, ChevronLeft, ChevronRight, X, AlertTriangle, ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react';
import axios from 'axios';

// Modal Component
const Modal = ({ isOpen, onClose, onConfirm, title, message, confirmText = 'Delete', confirmVariant = 'danger' }) => {
  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      animation: 'fadeIn 0.2s ease-out'
    }}>
      <div style={{
        backgroundColor: 'var(--background-card)',
        borderRadius: '12px',
        padding: '24px',
        maxWidth: '400px',
        width: '90%',
        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
        animation: 'slideUp 0.2s ease-out'
      }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px' }}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0
          }}>
            <AlertTriangle size={20} style={{ color: 'var(--danger-color)' }} />
          </div>
          <div style={{ flex: 1 }}>
            <h3 style={{ margin: '0 0 8px 0', fontSize: '18px', fontWeight: '600' }}>{title}</h3>
            <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '14px', lineHeight: '1.5' }}>{message}</p>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: '4px',
              color: 'var(--text-secondary)'
            }}
          >
            <X size={20} />
          </button>
        </div>
        <div style={{ display: 'flex', gap: '12px', marginTop: '24px', justifyContent: 'flex-end' }}>
          <button
            onClick={onClose}
            className="btn btn-secondary"
            style={{ minWidth: '80px' }}
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className={`btn ${confirmVariant === 'danger' ? 'btn-danger' : 'btn-primary'}`}
            style={{ minWidth: '80px' }}
          >
            {confirmText}
          </button>
        </div>
      </div>
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes slideUp {
          from { transform: translateY(20px); opacity: 0; }
          to { transform: translateY(0); opacity: 1; }
        }
      `}</style>
    </div>
  );
};

const History = () => {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  
  // Sorting state
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'desc' });
  
  // Modal states
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [clearAllModalOpen, setClearAllModalOpen] = useState(false);
  const [itemToDelete, setItemToDelete] = useState(null);

  useEffect(() => {
    fetchResults();
  }, [currentPage, filterStatus]);

  const fetchResults = async () => {
    setLoading(true);
    try {
      const response = await axios.get('http://localhost:5000/api/results', {
        params: { page: currentPage, status: filterStatus }
      });
      setResults(response.data.results || []);
      setTotalPages(response.data.total_pages || 1);
      setTotal(response.data.total || 0);
    } catch (error) {
      console.error('Error fetching results:', error);
      setResults([]);
      setTotalPages(1);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  };

  // Filter results based on search term
  const filteredResults = results.filter(r => 
    r.file_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    r.result.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Sorting function
  const sortedResults = [...filteredResults].sort((a, b) => {
    if (!sortConfig.key) return 0;
    
    let aValue, bValue;
    
    if (sortConfig.key === 'confidence') {
      aValue = a.confidence;
      bValue = b.confidence;
    } else if (sortConfig.key === 'date') {
      aValue = new Date(a.created_at).getTime();
      bValue = new Date(b.created_at).getTime();
    } else if (sortConfig.key === 'status') {
      // Sort by is_normal (true first for desc, false first for asc)
      aValue = a.is_normal ? 1 : 0;
      bValue = b.is_normal ? 1 : 0;
    } else {
      return 0;
    }
    
    if (aValue < bValue) return sortConfig.direction === 'asc' ? -1 : 1;
    if (aValue > bValue) return sortConfig.direction === 'asc' ? 1 : -1;
    return 0;
  });

  // Sort handler
  const handleSort = (key) => {
    setSortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'desc' ? 'asc' : 'desc'
    }));
  };

  // Get sort icon
  const getSortIcon = (columnKey) => {
    if (sortConfig.key !== columnKey) {
      return <ArrowUpDown size={14} style={{ opacity: 0.5 }} />;
    }
    return sortConfig.direction === 'asc' ? 
      <ArrowUp size={14} style={{ color: 'var(--primary-color)' }} /> : 
      <ArrowDown size={14} style={{ color: 'var(--primary-color)' }} />;
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Delete single result - opens modal
  const handleDeleteClick = (id) => {
    setItemToDelete(id);
    setDeleteModalOpen(true);
  };

  // Confirm delete
  const confirmDelete = async () => {
    if (!itemToDelete) return;
    
    try {
      await axios.delete(`http://localhost:5000/api/results/${itemToDelete}`);
      fetchResults();
    } catch (error) {
      console.error('Error deleting result:', error);
      setResults(results.filter(r => r.id !== itemToDelete));
    } finally {
      setDeleteModalOpen(false);
      setItemToDelete(null);
    }
  };

  // Clear all - opens modal
  const handleClearAllClick = () => {
    setClearAllModalOpen(true);
  };

  // Confirm clear all
  const confirmClearAll = async () => {
    try {
      await axios.delete('http://localhost:5000/api/results');
      fetchResults();
    } catch (error) {
      console.error('Error clearing results:', error);
      setResults([]);
    } finally {
      setClearAllModalOpen(false);
    }
  };

  return (
    <div className="history-page">
      {/* Delete Confirmation Modal */}
      <Modal
        isOpen={deleteModalOpen}
        onClose={() => setDeleteModalOpen(false)}
        onConfirm={confirmDelete}
        title="Delete Result"
        message="Are you sure you want to delete this analysis result? This action cannot be undone."
        confirmText="Delete"
        confirmVariant="danger"
      />

      {/* Clear All Confirmation Modal */}
      <Modal
        isOpen={clearAllModalOpen}
        onClose={() => setClearAllModalOpen(false)}
        onConfirm={confirmClearAll}
        title="Clear All Results"
        message="Are you sure you want to delete ALL analysis results? This action cannot be undone."
        confirmText="Clear All"
        confirmVariant="danger"
      />

      <div className="page-header">
        <div>
          <h1>Analysis History</h1>
          <p className="text-secondary">View all previous ECG analyses ({total} total)</p>
        </div>
        {total > 0 && (
          <button 
            className="btn btn-secondary"
            onClick={handleClearAllClick}
            style={{ color: 'var(--danger-color)' }}
          >
            <Trash2 size={16} />
            Clear All
          </button>
        )}
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
        ) : sortedResults.length > 0 ? (
          <>
            <div className="table-container">
              <table className="table">
                <thead>
                  <tr>
                    <th>File Name</th>
                    <th>Result</th>
                    <th 
                      onClick={() => handleSort('confidence')}
                      style={{ cursor: 'pointer', userSelect: 'none' }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        Confidence {getSortIcon('confidence')}
                      </div>
                    </th>
                    <th 
                      onClick={() => handleSort('date')}
                      style={{ cursor: 'pointer', userSelect: 'none' }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        Date {getSortIcon('date')}
                      </div>
                    </th>
                    <th 
                      onClick={() => handleSort('status')}
                      style={{ cursor: 'pointer', userSelect: 'none' }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        Status {getSortIcon('status')}
                      </div>
                    </th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedResults.map((result) => (
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
                          <button 
                            className="btn btn-sm btn-secondary"
                            onClick={() => handleDeleteClick(result.id)}
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
            <p>Upload an ECG file to start analysis</p>
            <Link to="/upload" className="btn btn-primary" style={{ marginTop: '16px' }}>
              Upload ECG
            </Link>
          </div>
        )}
      </div>
    </div>
  );
};

export default History;
