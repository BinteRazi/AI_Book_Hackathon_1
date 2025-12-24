import React from 'react';
import CookieConsentBanner from './CookieConsentBanner';

interface LayoutWrapperProps {
  children: React.ReactNode;
}

const LayoutWrapper: React.FC<LayoutWrapperProps> = ({ children }) => {
  return (
    <>
      {children}
      <CookieConsentBanner />
    </>
  );
};

export default LayoutWrapper;