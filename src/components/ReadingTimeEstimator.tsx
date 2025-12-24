import React from 'react';
import clsx from 'clsx';
import styles from './ReadingTimeEstimator.module.css';

interface ReadingTimeEstimatorProps {
  text: string;
  wordsPerMinute?: number;
  className?: string;
  icon?: boolean;
}

const ReadingTimeEstimator: React.FC<ReadingTimeEstimatorProps> = ({
  text,
  wordsPerMinute = 200, // Average reading speed
  className = '',
  icon = true
}) => {
  // Count words in the text
  const countWords = (text: string): number => {
    // Remove HTML tags and extra whitespace, then count words
    const cleanText = text.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
    if (!cleanText) return 0;
    return cleanText.split(/\s+/).length;
  };

  const wordCount = countWords(text);
  const readingTime = Math.ceil(wordCount / wordsPerMinute);

  return (
    <div className={clsx(styles.readingTime, className)}>
      {icon && <span className={styles.icon}>⏱️</span>}
      <span className={styles.text}>
        {readingTime} min read
        <span className={styles.wordCount}> ({wordCount} words)</span>
      </span>
    </div>
  );
};

export default ReadingTimeEstimator;