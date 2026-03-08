import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Upload, History, Settings, Info } from 'lucide-react';

const Sidebar = ({ isOpen }) => {
  const navItems = [
    { path: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/upload', icon: Upload, label: 'Upload' },
    { path: '/history', icon: History, label: 'History' },
    { path: '/settings', icon: Settings, label: 'Settings' },
  ];

  return (
    <aside className={`sidebar ${isOpen ? 'open' : 'closed'}`}>
      <NavLink to="/home" className="sidebar-header">
        <div className="title">Arrhythmia Detector</div>
      </NavLink>
      
      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
          >
            <item.icon size={20} />
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <div style={{ padding: '6px', borderTop: '1px solid var(--border-color)' }}>
        <div style={{ 
          padding: '6px', 
          background: 'var(--background-secondary)', 
          borderRadius: '8px',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <Info size={16} style={{ color: 'var(--text-secondary)' }} />
          <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
            Arrhythmia Detector v1.0.0
          </span>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;

