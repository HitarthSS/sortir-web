import React, { useState } from 'react';
import { Video, Link, Play } from 'lucide-react';

const VideoInput = ({ onAddVideo }) => {
  const [videoUrl, setVideoUrl] = useState('');
  const [isValid, setIsValid] = useState(false);

  const validateUrl = (url) => {
    // Basic URL validation - can be enhanced for specific platforms
    const urlPattern = /^https?:\/\/.+/;
    return urlPattern.test(url.trim());
  };

  const handleUrlChange = (e) => {
    const url = e.target.value;
    setVideoUrl(url);
    setIsValid(validateUrl(url));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (isValid) {
      onAddVideo(videoUrl.trim());
    }
  };

  const handlePaste = (e) => {
    const pastedText = e.clipboardData.getData('text');
    if (validateUrl(pastedText)) {
      setVideoUrl(pastedText);
      setIsValid(true);
    }
  };

  return (
    <div className="video-input">
      <div className="input-container">
        <div className="input-header">
          <Video size={24} />
          <h2>Paste Video Link</h2>
        </div>
        
        <p className="input-description">
          Share a video from TikTok, YouTube, or any platform and paste the link here
        </p>
        
        <form onSubmit={handleSubmit} className="url-form">
          <div className="url-input-wrapper">
            <Link size={20} className="link-icon" />
            <input
              type="url"
              value={videoUrl}
              onChange={handleUrlChange}
              onPaste={handlePaste}
              placeholder="https://www.tiktok.com/@user/video/1234567890"
              className="url-input"
              autoFocus
            />
          </div>
          
          <button
            type="submit"
            disabled={!isValid}
            className="process-button"
          >
            <Play size={20} />
            Process Video
          </button>
        </form>
        
        <div className="supported-platforms">
          <h3>Supported Platforms:</h3>
          <ul>
            <li>🎵 TikTok</li>
            <li>📺 YouTube</li>
            <li>📱 Instagram Reels</li>
            <li>🎬 Any video platform</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default VideoInput; 