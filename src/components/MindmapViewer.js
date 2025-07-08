import React, { useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

const MindmapViewer = ({ data, searchTerm = '', onReset, onAddVideo }) => {
  const [selectedNode, setSelectedNode] = useState(null);

  // Node color mapping
  const getNodeColor = (node) => {
    switch (node.type || node.data?.type) {
      case 'root': return '#2563eb';
      case 'Category':
      case 'category': return '#7c3aed';
      case 'Subcategory':
      case 'subcategory': return '#db2777';
      case 'Cluster':
      case 'cluster': return '#f59e42';
      case 'Video':
      case 'video': return '#059669';
      default: return '#64748b';
    }
  };

  // Node size mapping
  const getNodeSize = (node) => {
    switch (node.type || node.data?.type) {
      case 'root': return 14;
      case 'Category':
      case 'category': return 10;
      case 'Subcategory':
      case 'subcategory': return 8;
      case 'Cluster':
      case 'cluster': return 6;
      case 'Video':
      case 'video': return 5;
      default: return 4;
    }
  };

  // Filter nodes based on search
  const filteredData = React.useMemo(() => {
    if (!data || !Array.isArray(data.nodes) || !Array.isArray(data.edges)) return { nodes: [], edges: [] };
    if (!searchTerm) return data;
    const filteredNodes = (data.nodes || []).filter(node => {
      const label = node.label || node.data?.label || '';
      const summary = node.data?.summary || '';
      const tips = node.data?.tips || [];
      return (
        label.toLowerCase().includes(searchTerm.toLowerCase()) ||
        summary.toLowerCase().includes(searchTerm.toLowerCase()) ||
        tips.some(tip => tip.toLowerCase().includes(searchTerm.toLowerCase()))
      );
    });
    const nodeIds = new Set(filteredNodes.map(n => n.id));
    const filteredEdges = (data.edges || []).filter(edge => 
      nodeIds.has(edge.source) && nodeIds.has(edge.target)
    );
    return { nodes: filteredNodes, edges: filteredEdges };
  }, [data, searchTerm]);

  if (!data || !Array.isArray(data.nodes) || data.nodes.length === 0) {
    return (
      <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#a1a1aa', fontSize: 20 }}>
        No mindmap data available.
      </div>
    );
  }

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', background: '#18181b' }}>
      <ForceGraph2D
        graphData={{
          nodes: filteredData.nodes.map(node => ({
            ...node,
            id: node.id || `node-${Math.random()}`,
            label: node.label || node.data?.label || node.id,
          })),
          links: filteredData.edges.map(edge => ({
            ...edge,
            id: edge.id || `edge-${Math.random()}`,
            source: edge.source || 'unknown',
            target: edge.target || 'unknown',
          }))
        }}
        nodeLabel={node => node.label}
        nodeColor={getNodeColor}
        nodeVal={getNodeSize}
        linkColor={() => '#a1a1aa'}
        linkWidth={link => link.weight || 1}
        linkDirectionalParticles={0}
        onNodeClick={setSelectedNode}
        width={window.innerWidth}
        height={window.innerHeight - 56}
        backgroundColor="#18181b"
      />
      {selectedNode && (
        <div style={{ position: 'absolute', top: 80, right: 32, minWidth: 260, maxWidth: 340, background: '#23232b', color: '#fff', borderRadius: 10, boxShadow: '0 2px 16px #0008', padding: 20, zIndex: 10 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <div style={{ fontWeight: 600, fontSize: 18 }}>{selectedNode.label || selectedNode.data?.label}</div>
            <button onClick={() => setSelectedNode(null)} style={{ background: 'none', border: 'none', color: '#fff', fontSize: 18, cursor: 'pointer' }}>×</button>
          </div>
          <div style={{ fontSize: 14, opacity: 0.85 }}>
            {selectedNode.data?.summary && <div style={{ marginBottom: 8 }}><b>Summary:</b> <span>{selectedNode.data.summary}</span></div>}
            {selectedNode.data?.keyPoints && selectedNode.data.keyPoints.length > 0 && (
              <div style={{ marginBottom: 8 }}>
                <b>Key Points:</b>
                <ul style={{ margin: 0, paddingLeft: 18 }}>
                  {selectedNode.data.keyPoints.map((point, i) => <li key={i}>{point}</li>)}
                </ul>
              </div>
            )}
            {selectedNode.data?.tips && selectedNode.data.tips.length > 0 && (
              <div style={{ marginBottom: 8 }}>
                <b>Tips:</b>
                <ul style={{ margin: 0, paddingLeft: 18 }}>
                  {selectedNode.data.tips.map((tip, i) => <li key={i}>{tip}</li>)}
                </ul>
              </div>
            )}
            {selectedNode.data?.duration && <div><b>Duration:</b> {selectedNode.data.duration}</div>}
            {selectedNode.data?.platform && <div><b>Platform:</b> {selectedNode.data.platform}</div>}
            {selectedNode.data?.uploader && <div><b>Uploader:</b> {selectedNode.data.uploader}</div>}
            <div style={{ marginTop: 8, fontSize: 12, opacity: 0.7 }}><b>ID:</b> {selectedNode.id}</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MindmapViewer; 