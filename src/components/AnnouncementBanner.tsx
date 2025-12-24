import React, { useState, useEffect } from 'react';
import clsx from 'clsx';
import styles from './AnnouncementBanner.module.css';

interface Announcement {
  id: string;
  title: string;
  content: string;
  type: 'info' | 'warning' | 'success' | 'error';
  show: boolean;
  dismissible: boolean;
}

const AnnouncementBanner: React.FC = () => {
  const [announcements, setAnnouncements] = useState<Announcement[]>([
    {
      id: 'announcement-1',
      title: 'New Feature Available!',
      content: 'We just launched our new AI-powered chatbot. Check it out on our homepage!',
      type: 'success',
      show: true,
      dismissible: true,
    },
    // Add more announcements as needed
  ]);

  const [activeAnnouncement, setActiveAnnouncement] = useState<Announcement | null>(null);

  useEffect(() => {
    // Find the first announcement that should be shown
    const visibleAnnouncement = announcements.find(ann => ann.show);
    setActiveAnnouncement(visibleAnnouncement || null);
  }, [announcements]);

  const dismissAnnouncement = (id: string) => {
    // Update the announcement to not show anymore
    const updatedAnnouncements = announcements.map(ann =>
      ann.id === id ? { ...ann, show: false } : ann
    );
    setAnnouncements(updatedAnnouncements);

    // Also save to localStorage so it doesn't show again in future visits
    localStorage.setItem(`announcement-dismissed-${id}`, 'true');
  };

  if (!activeAnnouncement) {
    return null;
  }

  const bannerType = activeAnnouncement.type;
  const isDismissible = activeAnnouncement.dismissible;

  return (
    <div className={clsx(styles.announcementBanner, styles[bannerType])}>
      <div className={styles.announcementContent}>
        <div className={styles.announcementText}>
          <strong>{activeAnnouncement.title}</strong> {activeAnnouncement.content}
        </div>
        {isDismissible && (
          <button
            className={styles.dismissButton}
            onClick={() => dismissAnnouncement(activeAnnouncement.id)}
            aria-label="Dismiss announcement"
          >
            ✕
          </button>
        )}
      </div>
    </div>
  );
};

export default AnnouncementBanner;