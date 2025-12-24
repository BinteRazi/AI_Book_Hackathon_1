import React, { useState } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import BlogSearch from '../components/BlogSearch';
import Heading from '@theme/Heading';
import styles from './404.module.css';

function NotFoundPage(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  const [searchQuery, setSearchQuery] = useState('');

  const handleSearch = (query: string) => {
    setSearchQuery(query);
  };

  return (
    <Layout
      title="Page Not Found"
      description="The page you are looking for does not exist">
      <main className={styles.container}>
        <div className={styles.content}>
          <div className={styles.errorSection}>
            <div className={styles.errorCode}>404</div>
            <Heading as="h1" className={styles.title}>
              Page Not Found
            </Heading>
            <p className={styles.subtitle}>
              Oops! The page you're looking for doesn't exist or has been moved.
            </p>

            <div className={styles.searchSection}>
              <h3>Search our site:</h3>
              <BlogSearch />
            </div>

            <div className={styles.navigationSection}>
              <p>Here are some helpful links instead:</p>
              <div className={styles.linksGrid}>
                <Link to="/" className={clsx('button button--primary', styles.navButton)}>
                  Go Home
                </Link>
                <Link to="/docs/intro" className={clsx('button button--secondary', styles.navButton)}>
                  Documentation
                </Link>
                <Link to="/blog" className={clsx('button button--secondary', styles.navButton)}>
                  Blog
                </Link>
                <Link to="/contact" className={clsx('button button--secondary', styles.navButton)}>
                  Contact Us
                </Link>
              </div>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}

export default NotFoundPage;