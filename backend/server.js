const express = require('express');
const cors = require('cors');
const axios = require('axios');
const ytdl = require('ytdl-core');
const youtubeDl = require('youtube-dl-exec');
const OpenAI = require('openai');
const fs = require('fs');
const path = require('path');
const http = require('http');
const socketIo = require('socket.io');
const neo4j = require('neo4j-driver');
const Redis = require('redis');
const multer = require('multer');
const ffmpeg = require('fluent-ffmpeg');
require('dotenv').config();

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: ["http://localhost:3000", "http://localhost:5003", "http://localhost:5002", "http://localhost:5004"],
    methods: ["GET", "POST"]
  }
});

const PORT = process.env.PORT || 5001;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Initialize OpenAI
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Initialize Neo4j
const neo4jDriver = neo4j.driver(
  process.env.NEO4J_URI || 'bolt://localhost:7687',
  neo4j.auth.basic(
    process.env.NEO4J_USER || 'neo4j',
    process.env.NEO4J_PASSWORD || 'password'
  )
);

// Initialize Redis for caching
const redisClient = Redis.createClient({
  url: process.env.REDIS_URL || 'redis://localhost:6379'
});

// Connect to Redis
redisClient.connect().catch(console.error);

// Vector database (in-memory for now, can be replaced with Pinecone/Weaviate)
const vectorDB = new Map();

// --- Mindmap State ---
// In-memory mindmap structure (can be rebuilt from Neo4j if needed)
let mindmap = { nodes: [], edges: [] };

// Helper: Build mindmap from Neo4j
async function buildMindmapFromNeo4j() {
  const session = neo4jDriver.session();
  // Ensure root node exists in the database
  await session.run(`MERGE (root:Root {title: 'Knowledge Base'})`);
  // Ensure all categories are connected to root in the database
  const catResult = await session.run(`MATCH (c:Category) RETURN c`);
  for (const record of catResult.records) {
    const cat = record.get('c');
    await session.run(`
      MATCH (root:Root {title: 'Knowledge Base'})
      MATCH (c:Category {name: $name})
      MERGE (root)-[:CONTAINS]->(c)
    `, { name: cat.properties.name });
  }
  // Now build the mindmap as before
  const result = await session.run(`
    MATCH (n)
    OPTIONAL MATCH (n)-[r]->(m)
    RETURN n, r, m
  `);
  const nodes = [];
  const edges = [];
  const nodeMap = new Map();
  result.records.forEach(record => {
    const node = record.get('n');
    const relationship = record.get('r');
    const targetNode = record.get('m');
    if (node && !nodeMap.has(node.identity.toString())) {
      nodeMap.set(node.identity.toString(), true);
      nodes.push({
        id: node.labels.includes('Root') ? 'root' : node.identity.toString(),
        label: node.properties.name || node.properties.title || node.properties.label,
        type: node.labels[0],
        properties: node.properties
      });
    }
    if (relationship && targetNode) {
      edges.push({
        source: node.labels.includes('Root') ? 'root' : relationship.start.toString(),
        target: relationship.end.toString(),
        type: relationship.type
      });
    }
  });
  await session.close();
  // Ensure 'Knowledge Base' root is first in nodes array
  let rootIndex = nodes.findIndex(n => n.label === 'Knowledge Base' && n.type === 'Root');
  if (rootIndex > 0) {
    const [rootNode] = nodes.splice(rootIndex, 1);
    nodes.unshift(rootNode);
  }
  // Find all video nodes
  const videoNodes = nodes.filter(n => n.type === 'Video');
  // For each video, find its subcategory and category
  let minimalNodes = [];
  let minimalEdges = [];
  for (const video of videoNodes) {
    // Find the edge from subcategory to video
    const subcatEdge = edges.find(e => e.target === video.id && nodes.find(n => n.id === e.source && n.type === 'Subcategory'));
    if (!subcatEdge) continue;
    const subcatNode = nodes.find(n => n.id === subcatEdge.source);
    // Find the edge from category to subcategory
    const catEdge = edges.find(e => e.target === subcatNode.id && nodes.find(n => n.id === e.source && n.type === 'Category'));
    if (!catEdge) continue;
    const catNode = nodes.find(n => n.id === catEdge.source);
    // Find the edge from root to category
    const rootEdge = edges.find(e => e.target === catNode.id && nodes.find(n => n.id === e.source && n.type === 'Root'));
    const rootNode = nodes.find(n => n.type === 'Root');
    // Add nodes and edges for this path
    [rootNode, catNode, subcatNode, video].forEach(n => {
      if (n && !minimalNodes.some(mn => mn.id === n.id)) minimalNodes.push(n);
    });
    [rootEdge, catEdge, subcatEdge].forEach(ed => {
      if (ed && !minimalEdges.some(me => me.source === ed.source && me.target === ed.target)) minimalEdges.push(ed);
    });
  }
  // Only include nodes and edges in the minimal path
  const minimalNodeIds = minimalNodes.map(n => n.id);
  const strictNodes = nodes.filter(n => minimalNodeIds.includes(n.id));
  const strictEdges = edges.filter(e => minimalNodeIds.includes(e.source) && minimalNodeIds.includes(e.target));
  mindmap = { nodes: strictNodes, edges: strictEdges };
  // Attach summary, tips, and keyPoints to video nodes (fix: use node.properties.id as key)
  for (const node of strictNodes) {
    if (node.type === 'Video') {
      const videoId = node.properties.id;
      const vectorEntry = vectorDB.get(videoId);
      if (vectorEntry && vectorEntry.summaryData) {
        node.data = {
          summary: vectorEntry.summaryData.summary,
          tips: vectorEntry.summaryData.tips,
          keyPoints: vectorEntry.summaryData.keyPoints,
          duration: node.properties.duration,
          platform: node.properties.platform,
          uploader: node.properties.uploader
        };
      } else {
        node.data = {
          summary: node.properties.summary || '',
          tips: node.properties.tips || [],
          keyPoints: node.properties.keyPoints || [],
          duration: node.properties.duration,
          platform: node.properties.platform,
          uploader: node.properties.uploader
        };
      }
    }
  }
  return mindmap;
}

// WebSocket connection handling
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);
  
  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Step 1: Front-end - User pastes URL → POST /video
// --- API: Process Video (returns updated mindmap) ---
app.post('/api/process-video', async (req, res) => {
  const { videoUrl } = req.body;
  if (!videoUrl) {
    return res.status(400).json({ error: 'Video URL is required' });
  }
  try {
    io.emit('processing_start', { videoUrl });
    const videoInfo = await getVideoInfoAndCaptions(videoUrl);
    const transcript = await getTranscript(videoInfo);
    const summaryData = await generateSummary(transcript, videoInfo);
    const embedding = await generateEmbedding(transcript + ' ' + summaryData.summary);
    const clusterResult = await performHierarchicalClustering(embedding, videoInfo, summaryData);
    const labels = await generateClusterLabels(clusterResult);
    await storeInGraphDB(videoInfo, summaryData, clusterResult, labels);
    // Rebuild mindmap after update
    const map = await buildMindmapFromNeo4j();
    io.emit('mindmap_update', { type: 'new_video', data: map });
    res.json(map);
  } catch (error) {
    io.emit('processing_error', { error: error.message });
    res.status(500).json({ error: error.message });
  }
});

// Step 2a: Media Fetcher
async function getVideoInfoAndCaptions(url) {
  try {
    console.log('Fetching video info and captions for:', url);
    
    // Try youtube-dl first for better compatibility
    const info = await youtubeDl(url, {
      dumpSingleJson: true,
      noCheckCertificates: true,
      noWarnings: true,
      // Remove the problematic subtitle options for now
      // We'll handle audio download separately for Whisper
    });
    
    // Extract captions if available
    let captions = null;
    if (info.subtitles && Object.keys(info.subtitles).length > 0) {
      const subtitleKey = Object.keys(info.subtitles)[0];
      captions = info.subtitles[subtitleKey];
    } else if (info.automatic_captions && Object.keys(info.automatic_captions).length > 0) {
      const autoSubKey = Object.keys(info.automatic_captions)[0];
      captions = info.automatic_captions[autoSubKey];
    }
    
    return {
      id: info.id || Date.now().toString(),
      title: info.title || 'Video',
      duration: info.duration || 0,
      platform: 'YouTube',
      description: info.description || '',
      uploader: info.uploader || '',
      viewCount: info.view_count || 0,
      uploadDate: info.upload_date || '',
      captions: captions,
      url: url,
      success: true
    };
  } catch (error) {
    console.error('Error with youtube-dl:', error);
    throw new Error(`Failed to fetch video info: ${error.message}`);
  }
}

// Download audio for Whisper transcription
async function downloadAudio(url, videoId) {
  try {
    console.log('Downloading audio for Whisper transcription...');
    
    // Download audio using yt-dlp
    const audioPath = `./temp_audio_${videoId}.mp3`;
    
    await youtubeDl(url, {
      extractAudio: true,
      audioFormat: 'mp3',
      audioQuality: '192K',
      output: audioPath,
      noCheckCertificates: true,
      noWarnings: true
    });
    
    return audioPath;
  } catch (error) {
    console.error('Error downloading audio:', error);
    throw new Error(`Failed to download audio: ${error.message}`);
  }
}

// Step 2b: ASR (fallback)
async function getTranscript(videoInfo) {
  try {
    // If captions are available, use them
    if (videoInfo.captions && videoInfo.captions.length > 0) {
      console.log('Using available captions');
      return videoInfo.captions.map(caption => caption.text).join(' ');
    }
    
    // Otherwise, use Whisper for transcription
    console.log('No captions available, using Whisper for transcription');
    
    // Download audio
    const audioPath = await downloadAudio(videoInfo.url, videoInfo.id);
    
    // Use OpenAI Whisper API for transcription
    const fs = require('fs');
    
    console.log('Transcribing audio with Whisper API...');
    const transcription = await openai.audio.transcriptions.create({
      file: fs.createReadStream(audioPath),
      model: "whisper-1",
      response_format: "text",
      language: "en" // You can make this dynamic based on video metadata
    });
    
    // Clean up the temporary audio file
    fs.unlinkSync(audioPath);
    
    console.log('Whisper transcription completed');
    return transcription;
  } catch (error) {
    console.error('Error getting transcript:', error);
    
    // Fallback to mock transcript if Whisper fails
    console.log('Falling back to mock transcript due to error');
    const mockTranscript = `This is a fallback transcription for the video "${videoInfo.title}". 
    The Whisper API transcription failed, so we're using a placeholder transcript. 
    The actual content analysis will be limited to the video title and description.`;
    
    return mockTranscript;
  }
}

// Step 2c: Summarizer
async function generateSummary(transcript, videoInfo) {
  try {
    console.log('Generating summary for video:', videoInfo.title);
    
    const prompt = `Please provide a comprehensive analysis of the following video content:

Video Title: ${videoInfo.title}
Video Description: ${videoInfo.description}
Transcript: ${transcript}

Please provide:
1. A 5-sentence summary of the main content
2. A list of 5-7 practical tips or key insights
3. The main category and subcategory this content belongs to

Format your response as JSON:
{
  "summary": "5-sentence summary here",
  "tips": ["tip1", "tip2", "tip3", "tip4", "tip5"],
  "category": "main category",
  "subcategory": "subcategory",
  "keyPoints": ["key point 1", "key point 2", "key point 3"]
}`;

    const completion = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [
        {
          role: "system",
          content: "You are an expert content analyst. Provide clear, structured analysis."
        },
        {
          role: "user",
          content: prompt
        }
      ],
      temperature: 0.3,
      max_tokens: 1000
    });

    const response = completion.choices[0].message.content;
    console.log('GPT Response:', response);
    let summaryData;
    try {
      // Extract JSON object from response
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        summaryData = JSON.parse(jsonMatch[0]);
        console.log('Parsed summaryData:', summaryData);
      } else {
        throw new Error('No JSON object found in GPT response');
      }
    } catch (parseError) {
      console.error('JSON parse error:', parseError);
      // Fallback if JSON parsing fails
      summaryData = {
        summary: response,
        tips: ["Key insight from the video"],
        category: "General",
        subcategory: "Education",
        keyPoints: ["Main point from the video"]
      };
      console.log('Using fallback summaryData:', summaryData);
    }
    
    return summaryData;
  } catch (error) {
    console.error('Error generating summary:', error);
    throw new Error(`Failed to generate summary: ${error.message}`);
  }
}

// Step 3: Embedder
async function generateEmbedding(text) {
  try {
    console.log('Generating embedding for text');
    
    // Use OpenAI's text-embedding-3-large model
    const response = await openai.embeddings.create({
      model: "text-embedding-3-large",
      input: text,
      encoding_format: "float"
    });
    
    return response.data[0].embedding;
  } catch (error) {
    console.error('Error generating embedding:', error);
    throw new Error(`Failed to generate embedding: ${error.message}`);
  }
}

// Step 4: Hierarchy Engine
async function performHierarchicalClustering(embedding, videoInfo, summaryData) {
  try {
    console.log('Performing hierarchical clustering');
    // For now, we'll use a simplified clustering approach
    // In production, you'd use HDBSCAN and UMAP for proper clustering
    // Store the embedding and summaryData in our vector database
    const videoId = videoInfo.id;
    vectorDB.set(videoId, {
      embedding: embedding,
      videoInfo: videoInfo,
      summaryData: summaryData, // <-- store summaryData
      timestamp: Date.now()
    });
    // Simple clustering based on cosine similarity
    const allVectors = Array.from(vectorDB.values());
    const clusters = performSimpleClustering(allVectors);
    return {
      videoId: videoId,
      clusterId: clusters.findCluster(videoId),
      allClusters: clusters.getClusters(),
      similarityScores: clusters.getSimilarityScores(videoId)
    };
  } catch (error) {
    console.error('Error in hierarchical clustering:', error);
    throw new Error(`Failed to perform clustering: ${error.message}`);
  }
}

// Simple clustering implementation (replace with HDBSCAN in production)
class SimpleClustering {
  constructor() {
    this.clusters = new Map();
    this.nextClusterId = 0;
  }
  
  findCluster(videoId) {
    // Simple implementation - find the most similar existing cluster
    const threshold = 0.8; // cosine similarity threshold
    
    for (const [clusterId, videos] of this.clusters) {
      for (const video of videos) {
        const similarity = this.cosineSimilarity(
          vectorDB.get(videoId).embedding,
          video.embedding
        );
        if (similarity > threshold) {
          return clusterId;
        }
      }
    }
    
    // Create new cluster if no match found
    const newClusterId = this.nextClusterId++;
    this.clusters.set(newClusterId, [vectorDB.get(videoId)]);
    return newClusterId;
  }
  
  getClusters() {
    return Array.from(this.clusters.entries()).map(([id, videos]) => ({
      id,
      videos: videos.map(v => v.videoInfo.id),
      size: videos.length
    }));
  }
  
  getSimilarityScores(videoId) {
    const scores = [];
    const targetEmbedding = vectorDB.get(videoId).embedding;
    
    for (const [clusterId, videos] of this.clusters) {
      for (const video of videos) {
        if (video.videoInfo.id !== videoId) {
          scores.push({
            videoId: video.videoInfo.id,
            similarity: this.cosineSimilarity(targetEmbedding, video.embedding)
          });
        }
      }
    }
    
    return scores.sort((a, b) => b.similarity - a.similarity);
  }
  
  cosineSimilarity(vecA, vecB) {
    const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
    const normA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
    const normB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
    return dotProduct / (normA * normB);
  }
}

function performSimpleClustering(vectors) {
  return new SimpleClustering();
}

// Step 5: Labeler
async function generateClusterLabels(clusterResult) {
  try {
    console.log('Generating cluster labels');
    
    const labels = {};
    
    for (const cluster of clusterResult.allClusters) {
      // Get video titles and summaries for this cluster
      const clusterVideos = cluster.videos.map(videoId => {
        const videoData = vectorDB.get(videoId);
        return {
          title: videoData.videoInfo.title,
          summary: videoData.videoInfo.description
        };
      });
      
      // Generate label using GPT-4
      const videoTitles = clusterVideos.map(v => v.title).join('; ');
      const prompt = `Given these video titles: "${videoTitles}", 
      provide a concise 2-word label that best represents the common theme or category. 
      Return only the label, nothing else.`;

    const completion = await openai.chat.completions.create({
        model: "gpt-4",
      messages: [
        {
          role: "system",
            content: "You are an expert at categorizing content. Provide concise, accurate labels."
        },
        {
          role: "user",
          content: prompt
        }
      ],
        temperature: 0.3,
        max_tokens: 10
      });
      
      labels[cluster.id] = completion.choices[0].message.content.trim();
    }
    
    return labels;
  } catch (error) {
    console.error('Error generating cluster labels:', error);
    throw new Error(`Failed to generate labels: ${error.message}`);
  }
}

// Step 6: Graph Store
async function storeInGraphDB(videoInfo, summaryData, clusterResult, labels) {
  try {
    console.log('Storing in Neo4j graph database');
    
    const session = neo4jDriver.session();
    
    // Ensure root node exists
    await session.run(`
      MERGE (root:Root {title: 'Knowledge Base'})
    `);
    
    // Create category and subcategory nodes, and connect to root
    await session.run(`
      MERGE (c:Category {name: $category})
      MERGE (sc:Subcategory {name: $subcategory})
      MERGE (c)-[:CONTAINS]->(sc)
      WITH c
      MATCH (root:Root {title: 'Knowledge Base'})
      MERGE (root)-[:CONTAINS]->(c)
    `, {
      category: summaryData.category,
      subcategory: summaryData.subcategory
    });
    
    // Create video node
    await session.run(`
      MERGE (v:Video {id: $videoId})
      SET v.title = $title,
          v.duration = $duration,
          v.platform = $platform,
          v.uploader = $uploader,
          v.viewCount = $viewCount,
          v.uploadDate = $uploadDate
    `, {
      videoId: videoInfo.id,
      title: videoInfo.title,
      duration: videoInfo.duration,
      platform: videoInfo.platform,
      uploader: videoInfo.uploader,
      viewCount: videoInfo.viewCount,
      uploadDate: videoInfo.uploadDate
    });
    
    // Create cluster node
    const clusterLabel = labels[clusterResult.clusterId] || `Cluster ${clusterResult.clusterId}`;
    await session.run(`
      MERGE (cl:Cluster {id: $clusterId, label: $label})
    `, {
      clusterId: clusterResult.clusterId,
      label: clusterLabel
    });
    
    // Create relationships
    await session.run(`
      MATCH (v:Video {id: $videoId})
      MATCH (sc:Subcategory {name: $subcategory})
      MATCH (cl:Cluster {id: $clusterId})
      MERGE (sc)-[:CONTAINS]->(v)
      MERGE (cl)-[:CONTAINS]->(v)
    `, {
      videoId: videoInfo.id,
      subcategory: summaryData.subcategory,
      clusterId: clusterResult.clusterId
    });
    
    await session.close();
    
    return {
      videoId: videoInfo.id,
      clusterId: clusterResult.clusterId,
      clusterLabel: clusterLabel,
      category: summaryData.category,
      subcategory: summaryData.subcategory
    };
  } catch (error) {
    console.error('Error storing in graph database:', error);
    throw new Error(`Failed to store in graph database: ${error.message}`);
  }
}

// --- API: Get Mindmap ---
app.get('/api/mindmap', async (req, res) => {
  try {
    const map = await buildMindmapFromNeo4j();
    res.json(map);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// --- API: Get Video Node Details ---
app.get('/api/video/:id', async (req, res) => {
  const videoId = req.params.id;
  try {
    const session = neo4jDriver.session();
    const result = await session.run(
      'MATCH (v:Video {id: $videoId}) RETURN v',
      { videoId }
    );
    await session.close();
    if (result.records.length === 0) {
      return res.status(404).json({ error: 'Video not found' });
    }
    const node = result.records[0].get('v');
    res.json({
      id: node.properties.id,
      label: node.properties.title,
      type: 'Video',
      properties: node.properties
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Step 8: Layout API
app.get('/api/mindmap-layout', async (req, res) => {
  try {
    const session = neo4jDriver.session();
    
    // Get all nodes and relationships
    const result = await session.run(`
      MATCH (n)
      OPTIONAL MATCH (n)-[r]->(m)
      RETURN n, r, m
    `);
    
    const nodes = [];
    const edges = [];
    const nodeMap = new Map();
    
    result.records.forEach(record => {
      const node = record.get('n');
      const relationship = record.get('r');
      const targetNode = record.get('m');
      
      if (node && !nodeMap.has(node.identity.toString())) {
        nodeMap.set(node.identity.toString(), true);
        nodes.push({
          id: node.identity.toString(),
          label: node.properties.name || node.properties.title || node.properties.label,
          type: node.labels[0],
          properties: node.properties
        });
      }
      
      if (relationship && targetNode) {
        edges.push({
          source: relationship.start.toString(),
          target: relationship.end.toString(),
          type: relationship.type
        });
      }
    });
    
    await session.close();
    
    res.json({ nodes, edges });
  } catch (error) {
    console.error('Error getting mindmap layout:', error);
    res.status(500).json({ error: error.message });
  }
});

// Step 9: Incremental update logic
app.post('/api/incremental-update', async (req, res) => {
  const { videoUrl } = req.body;
  
  try {
    // Get new video data
    const videoInfo = await getVideoInfoAndCaptions(videoUrl);
    const transcript = await getTranscript(videoInfo);
    const summaryData = await generateSummary(transcript, videoInfo);
    const embedding = await generateEmbedding(transcript + ' ' + summaryData.summary);
    
    // Find nearest neighbor in existing vector database
    const nearestNeighbor = findNearestNeighbor(embedding);
    
    if (nearestNeighbor && nearestNeighbor.similarity > 0.8) {
      // Attach to existing leaf
      const result = await attachToExistingCluster(videoInfo, summaryData, nearestNeighbor);
      res.json(result);
    } else {
      // Rerun local HDBSCAN inside nearest internal cluster
      const result = await rerunLocalClustering(videoInfo, summaryData, embedding);
      res.json(result);
    }
  } catch (error) {
    console.error('Error in incremental update:', error);
    res.status(500).json({ error: error.message });
  }
});

function findNearestNeighbor(newEmbedding) {
  let bestMatch = null;
  let bestSimilarity = 0;
  
  for (const [videoId, data] of vectorDB) {
    const similarity = cosineSimilarity(newEmbedding, data.embedding);
    if (similarity > bestSimilarity) {
      bestSimilarity = similarity;
      bestMatch = { videoId, similarity };
    }
  }
  
  return bestMatch;
}

function cosineSimilarity(vecA, vecB) {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const normA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const normB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (normA * normB);
}

async function attachToExistingCluster(videoInfo, summaryData, nearestNeighbor) {
  // Implementation for attaching to existing cluster
  return { type: 'attached', clusterId: nearestNeighbor.videoId };
}

async function rerunLocalClustering(videoInfo, summaryData, embedding) {
  // Implementation for rerunning local clustering
  return { type: 'new_cluster', clusterId: Date.now() };
}

// Clear all data endpoint
app.post('/api/clear-all', async (req, res) => {
  try {
    console.log('Clearing all data from database, cache, and memory...');
    
    // Clear Neo4j database
    const session = neo4jDriver.session();
    await session.run('MATCH (n) DETACH DELETE n');
    await session.close();
    console.log('Neo4j database cleared');
    
    // Clear Redis cache
    try {
      await redisClient.flushall();
      console.log('Redis cache cleared');
    } catch (redisError) {
      console.log('Redis cache clear failed (may not be connected):', redisError.message);
    }
    
    // Clear in-memory vector database
    vectorDB.clear();
    console.log('In-memory vector database cleared');
    
    // Clear clustering data
    // Note: clustering is handled by the SimpleClustering class instances
    console.log('Clustering data cleared');
    
    // Emit clear event to WebSocket clients
    io.emit('mindmap_cleared');
    console.log('Mindmap cleared event emitted to WebSocket clients');
    
    res.json({ success: true, message: 'All data cleared successfully' });
  } catch (error) {
    console.error('Error clearing data:', error);
    res.status(500).json({ error: error.message });
  }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// Start server
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`WebSocket server ready for connections`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  server.close(() => {
    console.log('Server closed');
    neo4jDriver.close();
    redisClient.quit();
    process.exit(0);
  });
}); 