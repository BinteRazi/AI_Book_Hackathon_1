import React from 'react';
import {DarkModeProvider} from './components/contexts/DarkModeContext';
import {PageLoadingProvider} from './contexts/PageLoadingContext';
import {ProgressBarProvider} from './contexts/ProgressBarContext';
import {BlogRatingProvider} from './contexts/BlogRatingContext';
import PageTransitionWrapper from './components/PageTransitionWrapper';
import CookieConsentBanner from './components/CookieConsentBanner';
import LiveChatWidget from './components/LiveChatWidget';
import AnnouncementBanner from './components/AnnouncementBanner';
import FloatingActionButton from './components/FloatingActionButton';
import NotificationSystem from './components/NotificationSystem';

export default function Root({children}) {
  const fabActions = [
    {
      id: 'chat',
      label: 'Live Chat',
      icon: '💬',
      onClick: () => {
        // Trigger live chat
        const chatButton = document.getElementById('live-chat-button');
        if (chatButton) {
          (chatButton as HTMLElement).click();
        }
      }
    },
    {
      id: 'contact',
      label: 'Contact Us',
      icon: '📧',
      onClick: () => {
        window.location.href = '/contact';
      }
    },
    {
      id: 'top',
      label: 'Back to Top',
      icon: '↑',
      onClick: () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    }
  ];

  return (
    <DarkModeProvider>
      <ProgressBarProvider>
        <PageLoadingProvider>
          <BlogRatingProvider>
            <NotificationSystem>
              <AnnouncementBanner />
              <PageTransitionWrapper>
                {children}
              </PageTransitionWrapper>
              <CookieConsentBanner />
              <LiveChatWidget />
              <FloatingActionButton actions={fabActions} />
            </NotificationSystem>
          </BlogRatingProvider>
        </PageLoadingProvider>
      </ProgressBarProvider>
    </DarkModeProvider>
  );
}