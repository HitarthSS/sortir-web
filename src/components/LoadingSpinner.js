import React from 'react';
import { Loader2 } from 'lucide-react';

const LoadingSpinner = ({ message = 'Processing...' }) => {
  return (
    <div className="loading-spinner">
      <div className="spinner-container">
        <Loader2 size={48} className="spinner" />
        <h3>{message}</h3>
        <p>This may take a few moments...</p>
      </div>
    </div>
  );
};

export default LoadingSpinner; 