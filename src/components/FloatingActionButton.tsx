import React, { useState } from 'react';
import clsx from 'clsx';
import styles from './FloatingActionButton.module.css';

interface ActionItem {
  id: string;
  label: string;
  icon: string;
  onClick: () => void;
  color?: string;
}

interface FloatingActionButtonProps {
  actions: ActionItem[];
  mainIcon?: string;
  position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left';
  color?: string;
  className?: string;
}

const FloatingActionButton: React.FC<FloatingActionButtonProps> = ({
  actions,
  mainIcon = '+',
  position = 'bottom-right',
  color = 'var(--ifm-color-primary)',
  className = ''
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleMenu = () => {
    setIsOpen(!isOpen);
  };

  const handleClick = (action: ActionItem) => {
    action.onClick();
    setIsOpen(false); // Close the menu after clicking an action
  };

  const positionClasses = {
    'bottom-right': styles.bottomRight,
    'bottom-left': styles.bottomLeft,
    'top-right': styles.topRight,
    'top-left': styles.topLeft,
  };

  const fabStyle = {
    backgroundColor: color,
  };

  return (
    <div
      className={clsx(
        styles.fabContainer,
        positionClasses[position],
        className
      )}
    >
      <div className={clsx(styles.fabMenu, { [styles.open]: isOpen })}>
        {actions.map((action, index) => (
          <div
            key={action.id}
            className={clsx(
              styles.fabAction,
              styles[`action-${index + 1}`]
            )}
            style={{ '--delay': `${index * 50}ms` } as React.CSSProperties}
          >
            <button
              className={clsx(styles.fabActionBtn)}
              onClick={() => handleClick(action)}
              aria-label={action.label}
              title={action.label}
            >
              {action.icon}
            </button>
            <span className={styles.fabActionLabel}>{action.label}</span>
          </div>
        ))}
      </div>

      <button
        className={clsx(styles.fabMainBtn, { [styles.open]: isOpen })}
        onClick={toggleMenu}
        aria-expanded={isOpen}
        aria-label={isOpen ? "Close actions menu" : "Open actions menu"}
        style={fabStyle}
      >
        {isOpen ? '✕' : mainIcon}
      </button>
    </div>
  );
};

export default FloatingActionButton;