import React from 'react';
import clsx from 'clsx';
import styles from './SocialMediaShare.module.css';

interface SocialMediaShareProps {
  url?: string;
  title?: string;
  description?: string;
}

export default function SocialMediaShare({
  url = typeof window !== 'undefined' ? window.location.href : '',
  title = typeof window !== 'undefined' ? document.title : '',
  description = ''
}: SocialMediaShareProps): JSX.Element {
  const shareData = {
    twitter: {
      url: `https://twitter.com/intent/tweet?url=${encodeURIComponent(url)}&text=${encodeURIComponent(title)}`,
      icon: '🐦',
      label: 'Twitter'
    },
    facebook: {
      url: `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(url)}`,
      icon: '📘',
      label: 'Facebook'
    },
    linkedin: {
      url: `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(url)}`,
      icon: '💼',
      label: 'LinkedIn'
    },
    reddit: {
      url: `https://www.reddit.com/submit?url=${encodeURIComponent(url)}&title=${encodeURIComponent(title)}`,
      icon: '🔺',
      label: 'Reddit'
    }
  };

  const handleShare = (platform: keyof typeof shareData, e: React.MouseEvent) => {
    e.preventDefault();
    window.open(
      shareData[platform].url,
      '_blank',
      'noopener,noreferrer,width=600,height=400'
    );
  };

  return (
    <div className={clsx(styles.socialShareContainer)}>
      <h4>Share this post</h4>
      <div className={styles.socialButtons}>
        {Object.entries(shareData).map(([platform, data]) => (
          <a
            key={platform}
            href={data.url}
            onClick={(e) => handleShare(platform as keyof typeof shareData, e)}
            className={clsx(styles.socialButton, styles[platform])}
            aria-label={`Share on ${data.label}`}
            rel="noopener noreferrer"
            target="_blank"
          >
            <span className={styles.icon}>{data.icon}</span>
            <span className={styles.label}>{data.label}</span>
          </a>
        ))}
      </div>
    </div>
  );
}