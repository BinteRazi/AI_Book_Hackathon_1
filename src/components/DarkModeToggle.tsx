import React from 'react';
import { useDarkMode } from '../contexts/DarkModeContext';
import clsx from 'clsx';
import styles from './DarkModeToggle.module.css';

const DarkModeToggle: React.FC = () => {
  const { darkMode, toggleDarkMode } = useDarkMode();

  return (
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
  );
};

export default DarkModeToggle;