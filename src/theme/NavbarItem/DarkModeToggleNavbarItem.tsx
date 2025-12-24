import React from 'react';
import NavbarItem from '@theme/NavbarItem';
import { useDarkMode } from '../../contexts/DarkModeContext';

interface DarkModeToggleNavbarItemProps {
  type: string;
  position: string;
}

const DarkModeToggleNavbarItem: React.FC<DarkModeToggleNavbarItemProps> = () => {
  const { darkMode, toggleDarkMode } = useDarkMode();

  return (
    <div className="navbar__item">
      <button
        onClick={toggleDarkMode}
        className="clean-btn navbar__toggle"
        aria-label={`Switch to ${darkMode ? 'light' : 'dark'} mode`}
        title={`Switch to ${darkMode ? 'light' : 'dark'} mode`}
        style={{
          width: '36px',
          height: '36px',
          borderRadius: '50%',
          border: 'none',
          background: 'transparent',
          cursor: 'pointer',
          fontSize: '1.2rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--ifm-navbar-link-color)',
        }}
      >
        <span>
          {darkMode ? '☀️' : '🌙'}
        </span>
      </button>
    </div>
  );
};

export default DarkModeToggleNavbarItem;