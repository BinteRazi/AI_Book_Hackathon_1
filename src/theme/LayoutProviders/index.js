import React from 'react';
import {DarkModeProvider} from '../../contexts/DarkModeContext';

export default function LayoutWrapper({children}) {
  return <DarkModeProvider>{children}</DarkModeProvider>;
}