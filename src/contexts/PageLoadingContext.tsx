import React, { createContext, useContext, useState, ReactNode } from 'react';
import LoadingSpinner from '../components/LoadingSpinner';

interface PageLoadingContextType {
  isLoading: boolean;
  showLoading: () => void;
  hideLoading: () => void;
}

const PageLoadingContext = createContext<PageLoadingContextType | undefined>(undefined);

export const PageLoadingProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [isLoading, setIsLoading] = useState(false);

  const showLoading = () => setIsLoading(true);
  const hideLoading = () => setIsLoading(false);

  return (
    <PageLoadingContext.Provider value={{ isLoading, showLoading, hideLoading }}>
      {children}
      {isLoading && (
        <div className="page-loading-overlay">
          <LoadingSpinner size="large" message="Loading page..." />
        </div>
      )}
    </PageLoadingContext.Provider>
  );
};

export const usePageLoading = (): PageLoadingContextType => {
  const context = useContext(PageLoadingContext);
  if (context === undefined) {
    throw new Error('usePageLoading must be used within a PageLoadingProvider');
  }
  return context;
};