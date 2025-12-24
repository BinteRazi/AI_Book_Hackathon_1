import React, { useState, useEffect } from 'react';
import clsx from 'clsx';
import styles from './TestimonialCarousel.module.css';

interface Testimonial {
  id: number;
  name: string;
  role: string;
  company: string;
  content: string;
  avatar?: string;
}

const testimonials: Testimonial[] = [
  {
    id: 1,
    name: "Sarah Johnson",
    role: "CEO",
    company: "Tech Innovations",
    content: "This product has completely transformed our workflow. The efficiency gains are incredible!"
  },
  {
    id: 2,
    name: "Michael Chen",
    role: "CTO",
    company: "Digital Solutions",
    content: "Outstanding quality and support. Our team has seen a 50% increase in productivity since implementation."
  },
  {
    id: 3,
    name: "Emily Rodriguez",
    role: "Product Manager",
    company: "Future Systems",
    content: "The user experience is top-notch. Our customers love the intuitive interface and robust features."
  },
  {
    id: 4,
    name: "David Wilson",
    role: "Lead Developer",
    company: "Code Masters",
    content: "As a developer, I appreciate the clean codebase and comprehensive documentation. Highly recommended!"
  },
  {
    id: 5,
    name: "Lisa Thompson",
    role: "Marketing Director",
    company: "Brand Builders",
    content: "Our conversion rates improved significantly after using this solution. The ROI was exceptional."
  }
];

export default function TestimonialCarousel(): JSX.Element {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isAutoPlaying, setIsAutoPlaying] = useState(true);

  const goToPrevious = () => {
    const isFirstSlide = currentIndex === 0;
    const newIndex = isFirstSlide ? testimonials.length - 1 : currentIndex - 1;
    setCurrentIndex(newIndex);
  };

  const goToNext = () => {
    const isLastSlide = currentIndex === testimonials.length - 1;
    const newIndex = isLastSlide ? 0 : currentIndex + 1;
    setCurrentIndex(newIndex);
  };

  const goToSlide = (index: number) => {
    setCurrentIndex(index);
  };

  // Auto-advance carousel
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isAutoPlaying) {
      interval = setInterval(() => {
        goToNext();
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [isAutoPlaying, currentIndex]);

  // Pause auto-play when user interacts with carousel
  const handleMouseEnter = () => setIsAutoPlaying(false);
  const handleMouseLeave = () => setIsAutoPlaying(true);

  return (
    <div
      className={clsx(styles.testimonialCarousel)}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div className={styles.carouselHeader}>
        <h2>What Our Customers Say</h2>
        <p>Don't just take our word for it - hear from our satisfied customers</p>
      </div>

      <div className={styles.carouselContainer}>
        <button
          className={clsx(styles.navButton, styles.navButtonPrev)}
          onClick={goToPrevious}
          aria-label="Previous testimonial"
        >
          &#8249;
        </button>

        <div className={styles.testimonialSlide}>
          <div className={styles.testimonialContent}>
            <div className={styles.quoteIcon}>"</div>
            <p className={styles.testimonialText}>{testimonials[currentIndex].content}</p>
            <div className={styles.testimonialAuthor}>
              <div className={styles.authorInfo}>
                <h4>{testimonials[currentIndex].name}</h4>
                <p>{testimonials[currentIndex].role}, {testimonials[currentIndex].company}</p>
              </div>
            </div>
          </div>
        </div>

        <button
          className={clsx(styles.navButton, styles.navButtonNext)}
          onClick={goToNext}
          aria-label="Next testimonial"
        >
          &#8250;
        </button>
      </div>

      <div className={styles.indicators}>
        {testimonials.map((_, index) => (
          <button
            key={index}
            className={clsx(styles.indicator, {
              [styles.active]: index === currentIndex
            })}
            onClick={() => goToSlide(index)}
            aria-label={`Go to testimonial ${index + 1}`}
          />
        ))}
      </div>
    </div>
  );
}