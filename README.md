<!-- ```markdown -->
# RepoRecSys
GitHub Repository Recommendation System

A machine learning-based recommendation system for GitHub repositories, built with FastAPI backend and React frontend.

## Prerequisites

- Docker and Docker Compose
- Node.js (v16+) and npm (for frontend)
- Python 3.9+ (for local development/training)

## Data Preparation & Initial Training

Before starting the backend services, you need to download the dataset and train the initial model to generate the necessary artifacts.

### 1. Download Dataset

1.  Download the **GitRec** dataset from Kaggle:
    * [**Kaggle Link: gitrec-github-project-recommender-systems**](http://kaggle.com/datasets/mexwell/gitrec-github-project-recommender-systems)
2.  Extract the downloaded zip file.
3.  Create an `archive` folder inside `recsys_app/`:
    ```bash
    mkdir -p recsys_app/archive
    ```
4.  Move the following CSV files into `recsys_app/archive/`:
    * `train_balanced.csv`
    * `train_negative.csv`
    * `test_balanced.csv`
    * `test_negative.csv`

### 2. Train Initial Model

Navigate to the `recsys_app` directory and run the initial training script. This script processes the raw CSVs, trains the Two-Tower model, and saves the artifacts (model weights, mappings, scalers) required by the application.

1.  Navigate to the backend directory:
    ```bash
    cd recsys_app
    ```
2.  (Optional) Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  Run the training script:
    ```bash
    python3 init_training.py
    ```

**Output:**
Upon success, the script will create the following files in `recsys_app/artifacts/`:
* `best_twotower_model.pt`
* `mappings.pkl`
* `scalers.pkl`
* `item_features.csv`

---

## Quick Start

### 1. Backend Setup (Docker)

Ensure you are in the `recsys_app` directory:

```bash
cd recsys_app

```

#### Start Backend Services

Start all services (FastAPI app, GCS emulator, Pub/Sub emulator):

```bash
docker compose up --build -d

```

This will start:

* **FastAPI Backend**: `http://localhost:8000`
* **GCS Emulator**: `http://localhost:4443`
* **Pub/Sub Emulator**: `http://localhost:8085`

#### Seed Initial Data

After starting the services, seed the GCS emulator with the model artifacts you generated in the "Data Preparation" step:

```bash
docker compose exec app python3 seed_emulator.py

```

This uploads the local files from `artifacts/` to the emulated GCS bucket.

#### View Backend Logs

```bash
docker compose logs -f app

```

#### Stop Backend Services

```bash
docker compose down

```

### 2. Frontend Setup

Navigate to the `frontend` directory:

```bash
cd ../frontend

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

## Project Structure

```
RepoRecSys/
├── recsys_app/          # Backend FastAPI application
│   ├── main.py         # FastAPI app and endpoints
│   ├── model.py         # ML model definitions
│   ├── train_utils.py  # Training utilities
│   ├── init_training.py # Initial model training script
│   ├── seed_emulator.py # Script to upload artifacts to GCS emulator
│   ├── artifacts/       # Generated Model artifacts (gitignored)
│   ├── archive/         # Raw Training data from Kaggle (gitignored)
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

* **POST** `/recommend` - Get repository recommendations
```json
{
  "user_id": "123",
  "user_name": "optional_username",
  "top_k": 10
}

```



### Admin Endpoints

* **POST** `/system/ingest` - Trigger ingestion of new users from Pub/Sub
* **POST** `/system/train` - Trigger model training with new data
* **POST** `/system/reload` - Manually reload latest model from GCS
* **GET** `/system/status` - Get system status and logs

## Development Workflow

### Full Lifecycle Test

1. **Add a new user:**
* Use the frontend User View.
* Enter a new user ID (e.g., `99991`) and GitHub username.
* Get recommendations (this triggers the cold-start flow and queues the user).


2. **Ingest Data:**
* Go to Admin View.
* Click **"Trigger Ingest"**.
* This pulls the user from the queue, fetches their GitHub stars, and saves a training batch to GCS.


3. **Train Model:**
* Click **"Trigger Train"**.
* This downloads the base dataset + new batch, retrains the model, and uploads a new version to GCS.


4. **Reload Model:**
* Click **"Reload Model"** (or wait 10 mins for auto-reload).
* The server hot-swaps the model in memory.


5. **Verify:**
* Request recommendations for `99991` again. They should now be treated as a "Seen User" with personalized results.



## Environment Variables

Create a `.env` file in `recsys_app/` (optional, defaults are used if not set):

```env
GOOGLE_CLOUD_PROJECT=my-project-id
GITHUB_TOKEN=your_github_token_here  # Highly recommended for ingestion to avoid rate limits

```
