import React, { useState } from 'react';
import clsx from 'clsx';
import styles from './StarRating.module.css';

interface StarRatingProps {
  initialRating?: number;
  maxRating?: number;
  interactive?: boolean;
  size?: 'small' | 'medium' | 'large';
  onRatingChange?: (rating: number) => void;
  disabled?: boolean;
}

const StarRating: React.FC<StarRatingProps> = ({
  initialRating = 0,
  maxRating = 5,
  interactive = true,
  size = 'medium',
  onRatingChange,
  disabled = false
}) => {
  const [rating, setRating] = useState(initialRating);
  const [hoverRating, setHoverRating] = useState(0);

  const handleClick = (value: number) => {
    if (!interactive || disabled) return;

    setRating(value);
    if (onRatingChange) {
      onRatingChange(value);
    }
  };

  const handleMouseEnter = (value: number) => {
    if (!interactive || disabled) return;
    setHoverRating(value);
  };

  const handleMouseLeave = () => {
    if (!interactive || disabled) return;
    setHoverRating(0);
  };

  const sizeClasses = {
    small: styles.small,
    medium: styles.medium,
    large: styles.large,
  };

  return (
    <div className={clsx(styles.starRating, sizeClasses[size])}>
      {[...Array(maxRating)].map((_, index) => {
        const starValue = index + 1;
        const isFilled = starValue <= (hoverRating || rating);

        return (
          <button
            key={index}
            type="button"
            className={clsx(
              styles.star,
              isFilled ? styles.filled : styles.empty,
              interactive && !disabled && styles.interactive
            )}
            onClick={() => handleClick(starValue)}
            onMouseEnter={() => handleMouseEnter(starValue)}
            onMouseLeave={handleMouseLeave}
            disabled={!interactive || disabled}
            aria-label={`${starValue} star${starValue !== 1 ? 's' : ''}`}
            aria-pressed={isFilled}
          >
            ★
          </button>
        );
      })}
      <span className={styles.ratingText}>
        {rating.toFixed(1)}/{maxRating}
      </span>
    </div>
  );
};

export default StarRating;