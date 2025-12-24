import React, { useState, useEffect, JSX } from 'react';
import Link from '@docusaurus/Link';
import { useAllPluginInstancesData } from '@docusaurus/useGlobalData';
import clsx from 'clsx';
import styles from './BlogSearch.module.css';

interface BlogPost {
  metadata: {
    title: string;
    description: string;
    date: string;
    permalink: string;
    tags: { label: string; permalink: string }[];
  };
}

export default function BlogSearch(): JSX.Element {
  const [query, setQuery] = useState('');
  const [filteredPosts, setFilteredPosts] = useState<BlogPost[]>([]);
  const [showResults, setShowResults] = useState(false);

  // ✅ Correct way to access all blog posts in Docusaurus v3
  const blogData = useAllPluginInstancesData('docusaurus-plugin-content-blog');
  const allBlogPosts: BlogPost[] =
    (blogData?.[0] as any)?.blogPosts ?? [];

  useEffect(() => {
    if (!query.trim()) {
      setFilteredPosts([]);
      return;
    }

    const search = query.toLowerCase();

    const results = allBlogPosts.filter((post) => {
      const title = post.metadata.title.toLowerCase();
      const description = post.metadata.description?.toLowerCase() ?? '';
      const tags = post.metadata.tags
        .map((t) => t.label.toLowerCase())
        .join(' ');

      return (
        title.includes(search) ||
        description.includes(search) ||
        tags.includes(search)
      );
    });

    setFilteredPosts(results);
  }, [query, allBlogPosts]);

  const clearSearch = () => {
    setQuery('');
    setFilteredPosts([]);
    setShowResults(false);
  };

  return (
    <div className={styles.blogSearchContainer}>
      <div className={styles.searchWrapper}>
        <input
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setShowResults(e.target.value.length > 0);
          }}
          placeholder="Search blog posts..."
          className={clsx('form-control', styles.searchInput)}
        />

        {query && (
          <button
            onClick={clearSearch}
            className={styles.clearButton}
            aria-label="Clear search"
          >
            ✕
          </button>
        )}
      </div>

      {showResults && (
        <div className={styles.searchResults}>
          {filteredPosts.length > 0 ? (
            <ul className={styles.resultsList}>
              {filteredPosts.map((post, index) => (
                <li key={index} className={styles.resultItem}>
                  <Link to={post.metadata.permalink} className={styles.resultLink}>
                    <h4>{post.metadata.title}</h4>
                    <p>
                      {post.metadata.description?.slice(0, 100)}…
                    </p>
                  </Link>
                </li>
              ))}
            </ul>
          ) : (
            <div className={styles.noResults}>
              No blog posts found for "{query}"
            </div>
          )}
        </div>
      )}
    </div>
  );
}
