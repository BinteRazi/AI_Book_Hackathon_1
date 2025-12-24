import React, { useState, useEffect } from 'react';
import clsx from 'clsx';
import styles from './TableOfContents.module.css';

interface Heading {
  id: string;
  text: string;
  level: number;
}

interface TableOfContentsProps {
  className?: string;
  headings?: Heading[];
  title?: string;
}

const TableOfContents: React.FC<TableOfContentsProps> = ({
  className = '',
  title = 'Table of Contents',
  headings = []
}) => {
  const [tocHeadings, setTocHeadings] = useState<Heading[]>(headings);
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    // If headings are not provided, extract them from the page
    if (headings.length === 0) {
      const extractedHeadings = extractHeadingsFromPage();
      setTocHeadings(extractedHeadings);
    }
  }, [headings]);

  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY + 100; // Add offset for navbar

      // Find the heading that is currently in view
      const visibleHeading = tocHeadings
        .map(heading => {
          const element = document.getElementById(heading.id);
          if (!element) return null;

          const rect = element.getBoundingClientRect();
          const top = rect.top + window.scrollY;
          const bottom = top + rect.height;

          return { heading, top, bottom };
        })
        .filter(Boolean)
        .reverse() // Reverse to prioritize lower headings
        .find(item => scrollPosition >= item!.top && scrollPosition < item!.bottom);

      if (visibleHeading) {
        setActiveId(visibleHeading.heading.id);
      }
    };

    window.addEventListener('scroll', handleScroll);
    // Initial call to set active heading
    handleScroll();

    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, [tocHeadings]);

  const extractHeadingsFromPage = (): Heading[] => {
    const headings = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'));
    return headings
      .filter(heading => heading.id) // Only include headings with IDs
      .map(heading => ({
        id: heading.id,
        text: heading.textContent || '',
        level: parseInt(heading.tagName.charAt(1)) // Extract number from h1, h2, etc.
      }));
  };

  const handleClick = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      // Calculate offset for fixed navbar
      const offset = 100;
      const bodyRect = document.body.getBoundingClientRect().top;
      const elementRect = element.getBoundingClientRect().top;
      const elementPosition = elementRect - bodyRect;
      const offsetPosition = elementPosition - offset;

      window.scrollTo({
        top: offsetPosition,
        behavior: 'smooth'
      });
    }
  };

  if (tocHeadings.length === 0) {
    return null;
  }

  return (
    <nav className={clsx(styles.toc, className)} role="navigation" aria-labelledby="toc-title">
      <h3 id="toc-title" className={styles.tocTitle}>{title}</h3>
      <ul className={styles.tocList}>
        {tocHeadings.map((heading) => (
          <li
            key={heading.id}
            className={clsx(
              styles.tocItem,
              styles[`level-${heading.level}`],
              activeId === heading.id && styles.active
            )}
            style={{ paddingLeft: `${(heading.level - 2) * 16}px` }}
          >
            <a
              href={`#${heading.id}`}
              className={styles.tocLink}
              onClick={(e) => {
                e.preventDefault();
                handleClick(heading.id);
              }}
              aria-current={activeId === heading.id ? 'location' : undefined}
            >
              {heading.text}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
};

export default TableOfContents;