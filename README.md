# RepoRecSys
GitHub Repository Recommendation System

A machine learning-based recommendation system for GitHub repositories, built with FastAPI backend and React frontend.

## Prerequisites

- Docker and Docker Compose
- Node.js (v16+) and npm (for frontend)
- Python 3.9+ (for local development, optional)

## Quick Start

### 1. Backend Setup (Docker)

Navigate to the `recsys_app` directory:

```bash
cd recsys_app
```

#### Start Backend Services

Start all services (FastAPI app, GCS emulator, Pub/Sub emulator):

```bash
docker compose up --build -d
```

This will start:
- **FastAPI Backend**: `http://localhost:8000`
- **GCS Emulator**: `http://localhost:4443`
- **Pub/Sub Emulator**: `http://localhost:8085`

#### Seed Initial Data

After starting the services, seed the GCS emulator with initial model artifacts:

```bash
docker compose exec app python3 seed_emulator.py
```

This uploads:
- `best_twotower_model.pt` - Trained model
- `mappings.pkl` - User/item/language mappings
- `scalers.pkl` - Feature scalers
- `item_features.csv` - Item feature vectors

#### View Backend Logs

```bash
docker compose logs -f app
```

#### Stop Backend Services

```bash
docker compose down
```

#### Restart Backend Services

```bash
docker compose restart
```

### 2. Frontend Setup

Navigate to the `frontend` directory:

```bash
cd frontend
```

#### Install Dependencies

```bash
npm install
```

#### Start Development Server

```bash
npm run dev
```

The frontend will be available at `http://localhost:3000` (or the next available port).

#### Build for Production

```bash
npm run build
```

#### Preview Production Build

```bash
npm run preview
```

## Project Structure

```
RepoRecSys/
├── recsys_app/          # Backend FastAPI application
│   ├── main.py         # FastAPI app and endpoints
│   ├── model.py         # ML model definitions
│   ├── train_utils.py  # Training utilities
│   ├── init_training.py # Initial model training
│   ├── seed_emulator.py # Seed GCS emulator
│   ├── artifacts/      # Model artifacts (gitignored)
│   ├── archive/        # Training data (gitignored)
│   ├── docker-compose.yaml
│   └── Dockerfile
├── frontend/            # React + TypeScript frontend
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   └── services/
│   └── package.json
└── README.md
```

## API Endpoints

### User Endpoints

- **POST** `/recommend` - Get repository recommendations
  ```json
  {
    "user_id": "123",
    "user_name": "optional_username",
    "top_k": 10
  }
  ```

### Admin Endpoints

- **POST** `/system/ingest` - Trigger ingestion of new users from Pub/Sub
- **POST** `/system/train` - Trigger model training with new data
- **POST** `/system/reload` - Manually reload latest model from GCS
- **GET** `/system/status` - Get system status and logs

## Development Workflow

### Initial Setup

1. **Start backend services:**
   ```bash
   cd recsys_app
   docker compose up -d
   ```

2. **Seed emulator:**
   ```bash
   docker compose exec app python3 seed_emulator.py
   ```

3. **Start frontend:**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

### Testing the Pipeline

1. **Add a new user:**
   - Use the frontend User View
   - Enter a new user ID and optional GitHub username
   - Get recommendations (will trigger cold-start flow)

2. **Process new users:**
   - Go to Admin View
   - Click "Trigger Ingest" to process new users from Pub/Sub

3. **Train model:**
   - Click "Trigger Train" to train with new data
   - Wait for training to complete

4. **Reload model:**
   - Click "Reload Model" to load the newly trained model
   - Test recommendations again for the new user

5. **View logs:**
   - Click "Refresh Status" in Admin View to see system logs

## Environment Variables

Create a `.env` file in `recsys_app/` (optional, defaults are used if not set):

```env
GOOGLE_CLOUD_PROJECT=my-project-id
GITHUB_TOKEN=your_github_token_here  # Optional, for higher API rate limits
```
