import React, { createContext, useContext, useState, ReactNode } from 'react';
import ProgressBar from '../components/ProgressBar';

interface ProgressBarContextType {
  progress: number;
  showProgressBar: boolean;
  setProgress: (progress: number) => void;
  showProgress: () => void;
  hideProgress: () => void;
  startProgress: () => void;
  completeProgress: () => void;
}

const ProgressBarContext = createContext<ProgressBarContextType | undefined>(undefined);

export const ProgressBarProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [progress, setProgress] = useState(0);
  const [showProgressBar, setShowProgressBar] = useState(false);

  const showProgress = () => setShowProgressBar(true);
  const hideProgress = () => {
    setProgress(100);
    setTimeout(() => {
      setShowProgressBar(false);
      setProgress(0);
    }, 300); // Wait for animation to complete
  };

  const startProgress = () => {
    setShowProgressBar(true);
    setProgress(0);

    // Simulate progress
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) {
          clearInterval(interval);
          return 90;
        }
        return prev + 5;
      });
    }, 200);
  };

  const completeProgress = () => {
    setProgress(100);
    setTimeout(() => {
      setShowProgressBar(false);
      setProgress(0);
    }, 300);
  };

  const value = {
    progress,
    showProgressBar,
    setProgress,
    showProgress,
    hideProgress,
    startProgress,
    completeProgress
  };

  return (
    <ProgressBarContext.Provider value={value}>
      {children}
      {showProgressBar && (
        <div className="top-progress-bar">
          <ProgressBar
            progress={progress}
            show={showProgressBar}
            className="topProgressBar"
          />
        </div>
      )}
    </ProgressBarContext.Provider>
  );
};

export const useProgressBar = (): ProgressBarContextType => {
  const context = useContext(ProgressBarContext);
  if (context === undefined) {
    throw new Error('useProgressBar must be used within a ProgressBarProvider');
  }
  return context;
};