import React, { useEffect } from 'react';
import { useLocation } from '@docusaurus/router';
import { usePageLoading } from '../contexts/PageLoadingContext';

const PageTransitionWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { pathname } = useLocation();
  const { showLoading, hideLoading } = usePageLoading();

  useEffect(() => {
    // Show loading when route changes
    showLoading();

    // Simulate loading completion after a short delay
    // In a real implementation, this would be tied to actual page loading
    const timer = setTimeout(() => {
      hideLoading();
    }, 300); // Small delay to make transition visible

    return () => clearTimeout(timer);
  }, [pathname, showLoading, hideLoading]);

  return <>{children}</>;
};

export default PageTransitionWrapper;