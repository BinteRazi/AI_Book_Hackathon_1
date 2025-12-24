import React, { createContext, useContext, useState, ReactNode } from 'react';

interface BlogRatingContextType {
  getRating: (postId: string) => number;
  setRating: (postId: string, rating: number) => void;
  getAverageRating: (postId: string) => number;
  getRatingCount: (postId: string) => number;
}

const BlogRatingContext = createContext<BlogRatingContextType | undefined>(undefined);

export const BlogRatingProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [ratings, setRatings] = useState<Record<string, number[]>>({});

  const getRating = (postId: string): number => {
    const userRatings = ratings[postId] || [];
    return userRatings.length > 0 ? userRatings[userRatings.length - 1] : 0;
  };

  const setRating = (postId: string, rating: number): void => {
    setRatings(prev => ({
      ...prev,
      [postId]: [...(prev[postId] || []), rating]
    }));
  };

  const getAverageRating = (postId: string): number => {
    const postRatings = ratings[postId] || [];
    if (postRatings.length === 0) return 0;

    const sum = postRatings.reduce((acc, curr) => acc + curr, 0);
    return sum / postRatings.length;
  };

  const getRatingCount = (postId: string): number => {
    return ratings[postId]?.length || 0;
  };

  const value = {
    getRating,
    setRating,
    getAverageRating,
    getRatingCount
  };

  return (
    <BlogRatingContext.Provider value={value}>
      {children}
    </BlogRatingContext.Provider>
  );
};

export const useBlogRating = (): BlogRatingContextType => {
  const context = useContext(BlogRatingContext);
  if (context === undefined) {
    throw new Error('useBlogRating must be used within a BlogRatingProvider');
  }
  return context;
};