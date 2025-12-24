import React from 'react';
import { useBlogPost } from '@docusaurus/plugin-content-blog/client';
import BlogPostItem from '@theme-original/BlogPostItem';
import SocialMediaShare from '../components/SocialMediaShare';

export default function BlogPostItemWrapper(props) {
  const {isBlogPostPage} = useBlogPost();
  const {children} = props;

  return (
    <>
      <BlogPostItem {...props}>
        {children}
        {isBlogPostPage && (
          <div className="margin-vert--xl">
            <SocialMediaShare />
          </div>
        )}
      </BlogPostItem>
    </>
  );
}