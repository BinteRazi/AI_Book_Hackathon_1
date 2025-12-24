import React from 'react';
import clsx from 'clsx';
import styles from './ProgressBar.module.css';

interface ProgressBarProps {
  progress: number; // 0 to 100
  show?: boolean;
  height?: number;
  color?: string;
  backgroundColor?: string;
  className?: string;
}

const ProgressBar: React.FC<ProgressBarProps> = ({
  progress,
  show = true,
  height = 4,
  color = 'var(--ifm-color-primary)',
  backgroundColor = 'var(--ifm-color-emphasis-200)',
  className = ''
}) => {
  if (!show || progress < 0 || progress > 100) {
    return null;
  }

  const progressStyle = {
    width: `${progress}%`,
    height: `${height}px`,
    backgroundColor: color,
  };

  const containerStyle = {
    height: `${height}px`,
    backgroundColor: backgroundColor,
  };

  return (
    <div
      className={clsx(styles.progressBarContainer, className)}
      style={containerStyle}
      role="progressbar"
      aria-valuenow={progress}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div
        className={styles.progressBarFill}
        style={progressStyle}
      />
    </div>
  );
};

export default ProgressBar;