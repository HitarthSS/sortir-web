# Sortir Web

A video knowledge management system that processes YouTube/TikTok videos, extracts content, and organizes it into an interactive mindmap.

## Features
- **Backend:** Node.js/Express, Neo4j, Redis, OpenAI GPT-4, Whisper, yt-dlp
- **Frontend:** React, ForceGraph2D, WebSocket for real-time updates

---

## Quick Start

### 1. Clone the repository
```sh
git clone https://github.com/HitarthSS/sortir-web.git
cd sortir-web
```

### 2. Set up environment variables
- Copy `.env.example` to `.env` in both the root and `backend/` directories.
- Fill in your OpenAI, Neo4j, and Redis credentials in `backend/.env`.

#### `backend/.env.example` variables:
```
OPENAI_API_KEY=your_openai_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
REDIS_URL=redis://localhost:6379
PORT=5001
```

#### `.env.example` (frontend):
```
# Example: REACT_APP_API_URL=http://localhost:5001
```

### 3. Install dependencies
```sh
cd backend
npm install
cd ..
npm install
```

### 4. Start the backend
```sh
cd backend
npm start
```

### 5. Start the frontend (in a new terminal)
```sh
npm start
```

### 6. Open your browser
Go to [http://localhost:3000](http://localhost:3000) (or the port shown).

---

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## Issues
If you encounter bugs or have feature requests, please use the [GitHub Issues](https://github.com/HitarthSS/sortir-web/issues) page.

## License
[MIT](LICENSE)

---

**Note:**
- Do **not** commit your `.env` files. Use `.env.example` for sharing config structure.
- Backend runs on port 5001 by default; frontend on 3000.
- Requires Node.js, Neo4j, Redis, and an OpenAI API key.
