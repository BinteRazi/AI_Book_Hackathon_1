import React from 'react';
import clsx from 'clsx';
import { useDarkMode } from '../contexts/DarkModeContext';
import styles from './DarkModeToggleNavbarItem.module.css';

const DarkModeToggleNavbarItem: React.FC = () => {
  const { darkMode, toggleDarkMode } = useDarkMode();

  return (
    <div className={styles.navbarItem}>
      <button
        onClick={toggleDarkMode}
        className={clsx(styles.toggleButton, {
          [styles.dark]: darkMode,
          [styles.light]: !darkMode,
        })}
        aria-label={`Switch to ${darkMode ? 'light' : 'dark'} mode`}
        title={`Switch to ${darkMode ? 'light' : 'dark'} mode`}
      >
        <span className={styles.toggleIcon}>
          {darkMode ? '☀️' : '🌙'}
        </span>
      </button>
    </div>
  );
};

export default DarkModeToggleNavbarItem;