import React, { useState, useEffect, useCallback } from 'react';
import clsx from 'clsx';
import styles from './NotificationSystem.module.css';

interface Notification {
  id: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  duration?: number;
  onClose?: () => void;
}

interface NotificationSystemContextType {
  showNotification: (message: string, type?: 'info' | 'success' | 'warning' | 'error', duration?: number) => void;
}

const NotificationSystemContext = React.createContext<NotificationSystemContextType | undefined>(undefined);

export const useNotification = (): NotificationSystemContextType => {
  const context = React.useContext(NotificationSystemContext);
  if (!context) {
    throw new Error('useNotification must be used within a NotificationSystemProvider');
  }
  return context;
};

interface NotificationSystemProps {
  children: React.ReactNode;
}

const NotificationSystem: React.FC<NotificationSystemProps> = ({ children }) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const removeNotification = useCallback((id: string) => {
    setNotifications(prev => prev.filter(notification => notification.id !== id));
  }, []);

  const showNotification = useCallback((
    message: string,
    type: 'info' | 'success' | 'warning' | 'error' = 'info',
    duration: number = 5000
  ) => {
    const id = Math.random().toString(36).substr(2, 9);
    const notification: Notification = {
      id,
      message,
      type,
      duration,
    };

    setNotifications(prev => [...prev, notification]);

    // Auto-remove notification after duration
    if (duration > 0) {
      setTimeout(() => {
        removeNotification(id);
      }, duration);
    }

    return id;
  }, [removeNotification]);

  const contextValue = {
    showNotification
  };

  return (
    <NotificationSystemContext.Provider value={contextValue}>
      {children}
      <div className={styles.notificationContainer}>
        {notifications.map(notification => (
          <div
            key={notification.id}
            className={clsx(
              styles.notification,
              styles[notification.type]
            )}
            role="alert"
          >
            <div className={styles.notificationContent}>
              <span className={styles.notificationMessage}>
                {notification.message}
              </span>
              <button
                className={styles.notificationClose}
                onClick={() => {
                  removeNotification(notification.id);
                  if (notification.onClose) notification.onClose();
                }}
                aria-label="Close notification"
              >
                ✕
              </button>
            </div>
          </div>
        ))}
      </div>
    </NotificationSystemContext.Provider>
  );
};

export default NotificationSystem;