import React, { useState, useEffect } from 'react';
import clsx from 'clsx';
import styles from './CookieConsentBanner.module.css';

const CookieConsentBanner: React.FC = () => {
  const [showBanner, setShowBanner] = useState(false);

  useEffect(() => {
    // Check if user has already consented
    const consent = localStorage.getItem('cookieConsent');
    if (!consent) {
      setShowBanner(true);
    }
  }, []);

  const acceptCookies = () => {
    localStorage.setItem('cookieConsent', 'accepted');
    setShowBanner(false);
  };

  const declineCookies = () => {
    localStorage.setItem('cookieConsent', 'declined');
    setShowBanner(false);
  };

  if (!showBanner) {
    return null;
  }

  return (
    <div className={clsx(styles.cookieBanner)}>
      <div className={styles.cookieContent}>
        <div className={styles.cookieText}>
          <p>
            We use cookies to improve your experience on our website.
            By continuing to use our site, you consent to our use of cookies.
          </p>
        </div>
        <div className={styles.cookieButtons}>
          <button
            className={clsx(styles.button, styles.acceptButton)}
            onClick={acceptCookies}
          >
            Accept
          </button>
          <button
            className={clsx(styles.button, styles.declineButton)}
            onClick={declineCookies}
          >
            Decline
          </button>
        </div>
      </div>
    </div>
  );
};

export default CookieConsentBanner;