import React from 'react';
import clsx from 'clsx';
import styles from './PrintButton.module.css';

interface PrintButtonProps {
  className?: string;
  title?: string;
  size?: 'small' | 'medium' | 'large';
}

const PrintButton: React.FC<PrintButtonProps> = ({
  className = '',
  title = 'Print this page',
  size = 'medium'
}) => {
  const handlePrint = () => {
    window.print();
  };

  const sizeClasses = {
    small: styles.small,
    medium: styles.medium,
    large: styles.large,
  };

  return (
    <button
      className={clsx(
        styles.printButton,
        sizeClasses[size],
        className
      )}
      onClick={handlePrint}
      title={title}
      aria-label={title}
    >
      <span className={styles.printIcon}>⎙</span>
      <span className={styles.printText}>Print</span>
    </button>
  );
};

export default PrintButton;