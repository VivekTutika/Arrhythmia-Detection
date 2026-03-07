import { useLocation } from 'react-router-dom';
import { Menu, Sun, Moon, Bell, User } from 'lucide-react';

const pageTitles = {
  '/dashboard': 'Dashboard',
  '/upload': 'Upload ECG',
  '/history': 'Analysis History',
  '/settings': 'Settings',
};

const Header = ({ toggleSidebar, toggleTheme, theme }) => {
  const location = useLocation();
  const currentTitle = pageTitles[location.pathname] || 'Arrhythmia Detection';

  return (
    <header className="header">
      <div className="header-left">
        <button className="menu-toggle" onClick={toggleSidebar}>
          <Menu size={20} />
        </button>
        <h2 className="header-title">{currentTitle}</h2>
      </div>

      <div className="header-right">
        <button className="header-btn">
          <Bell size={20} />
        </button>
        
        <div className="theme-toggle" onClick={toggleTheme}>
          {theme === 'light' ? <Moon size={18} /> : <Sun size={18} />}
        </div>

        <button className="header-btn">
          <User size={20} />
        </button>
      </div>
    </header>
  );
};

export default Header;

