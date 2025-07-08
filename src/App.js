import React, { useState, useEffect } from 'react';
import { io } from 'socket.io-client';
import MindmapViewer from './components/MindmapViewer';
import LoadingSpinner from './components/LoadingSpinner';
import VideoInput from './components/VideoInput';

function App() {
  const [mindmapData, setMindmapData] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [showInput, setShowInput] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');

  // Fetch mindmap from backend on load
  useEffect(() => {
    fetchMindmap();
  }, []);

  // WebSocket for real-time updates
  useEffect(() => {
    const socket = io('http://localhost:5001');
    socket.on('mindmap_update', (data) => {
      setMindmapData(data.data);
      setIsProcessing(false);
      setShowInput(false);
    });
    socket.on('processing_start', () => setIsProcessing(true));
    socket.on('processing_error', (data) => {
      setError(data.error || 'Processing failed.');
      setIsProcessing(false);
    });
    socket.on('mindmap_cleared', () => setMindmapData(null));
    return () => socket.close();
  }, []);

  const fetchMindmap = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/mindmap');
      if (!response.ok) throw new Error('Backend unavailable');
      const data = await response.json();
      setMindmapData(data && data.nodes && data.edges ? data : null);
    } catch (err) {
      setError('Could not connect to backend.');
    }
  };

  const handleVideoProcess = async (videoUrl) => {
    setIsProcessing(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:5001/api/process-video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ videoUrl }),
      });
      if (!response.ok) throw new Error('Failed to process video');
      // Mindmap will update via WebSocket
    } catch (err) {
      setError('Failed to process video.');
      setIsProcessing(false);
    }
  };

  const clearMindmap = async () => {
    setIsProcessing(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:5001/api/clear-all', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!response.ok) throw new Error('Failed to clear data');
      setMindmapData(null);
      setIsProcessing(false);
    } catch (err) {
      setError('Failed to clear data.');
      setIsProcessing(false);
    }
  };

  // Minimal UI
  return (
    <div style={{ width: '100vw', height: '100vh', background: '#18181b', color: '#fff', fontFamily: 'Inter, sans-serif' }}>
      {isProcessing && <LoadingSpinner message="Processing video..." />}
      {error && (
        <div style={{ position: 'absolute', top: 24, left: 0, right: 0, textAlign: 'center', color: '#ef4444', fontWeight: 600, zIndex: 10 }}>{error}</div>
      )}
      {!mindmapData && !isProcessing && (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh' }}>
          <h1 style={{ fontSize: 32, fontWeight: 700, marginBottom: 8 }}>Sortir Mindmap</h1>
          <p style={{ color: '#a1a1aa', marginBottom: 32 }}>Paste a video URL to start building your knowledge graph.</p>
          <div style={{ minWidth: 340, maxWidth: 420, width: '100%' }}>
            <VideoInput onAddVideo={handleVideoProcess} />
          </div>
        </div>
      )}
      {mindmapData && mindmapData.nodes && mindmapData.edges && !isProcessing && (
        <div style={{ width: '100vw', height: '100vh', display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', alignItems: 'center', padding: '0.5rem 1.5rem', background: 'rgba(24,24,27,0.95)', borderBottom: '1px solid #27272a', zIndex: 2 }}>
            <h2 style={{ fontSize: '1.2rem', fontWeight: 600, margin: 0, marginRight: '2rem', color: '#fff' }}>Mindmap</h2>
            <input
              type="text"
              placeholder="Search nodes..."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              style={{ flex: 1, maxWidth: 300, marginRight: 16, padding: '0.4rem 0.8rem', borderRadius: 6, border: '1px solid #27272a', background: '#23232b', color: '#fff' }}
            />
            <button onClick={clearMindmap} style={{ marginRight: 12, background: '#ef4444', color: '#fff', border: 'none', borderRadius: 6, padding: '0.4rem 1rem', cursor: 'pointer' }}>Clear</button>
            <button onClick={() => setShowInput(v => !v)} style={{ background: '#2563eb', color: '#fff', border: 'none', borderRadius: 6, padding: '0.4rem 1rem', cursor: 'pointer' }}>Add Video</button>
          </div>
          {showInput && (
            <div style={{ background: '#23232b', padding: '1rem', borderBottom: '1px solid #27272a', display: 'flex', alignItems: 'center', gap: 12 }}>
              <VideoInput onAddVideo={url => { handleVideoProcess(url); setShowInput(false); }} />
              <button onClick={() => setShowInput(false)} style={{ marginLeft: 8, background: '#27272a', color: '#fff', border: 'none', borderRadius: 6, padding: '0.4rem 1rem', cursor: 'pointer' }}>Close</button>
            </div>
          )}
          <div style={{ flex: 1, minHeight: 0 }}>
            <MindmapViewer
              data={mindmapData}
              searchTerm={searchTerm}
              onReset={clearMindmap}
              onAddVideo={handleVideoProcess}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
