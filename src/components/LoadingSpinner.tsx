import React from 'react';
import clsx from 'clsx';
import styles from './LoadingSpinner.module.css';

interface LoadingSpinnerProps {
  size?: 'small' | 'medium' | 'large';
  show?: boolean;
  message?: string;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'medium',
  show = true,
  message = 'Loading...'
}) => {
  if (!show) {
    return null;
  }

  return (
    <div className={clsx(styles.spinnerContainer)}>
      <div className={clsx(styles.spinner, styles[size])}>
        <div className={styles.spinnerCircle}></div>
        <div className={styles.spinnerCircle}></div>
        <div className={styles.spinnerCircle}></div>
      </div>
      {message && (
        <p className={styles.spinnerMessage}>{message}</p>
      )}
    </div>
  );
};

export default LoadingSpinner;