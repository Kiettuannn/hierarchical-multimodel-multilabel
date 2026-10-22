# 0001 Separate Experiments from Production Architecture

## Status
Accepted

## Context
The repository was originally created for multimodal machine learning research, containing exploratory data analysis (EDA), offline modality extraction scripts, and hierarchical cascade training notebooks targeted specifically at Kaggle and Google Colab environments. These training and feature extraction pipelines require extensive GPU VRAM and hardcoded cloud paths, rendering them inoperable for direct execution on standard personal computers.

## Decision
We decouple the research artifacts from the production application code:
1. All legacy Kaggle/Colab preprocessing, training, and EDA pipelines are moved into an `experiments/` directory to preserve reproducible research without polluting the deployment footprint.
2. A production application structure is introduced with clean architecture separation:
   - `backend/`: FastAPI application separating domain entities, service use cases, infrastructure adapters (including mock and cloud-delegated inference services), and API controllers.
   - `frontend/`: Single-page application using React, TypeScript, Vite, and Tailwind CSS.
   - `docker-compose.yml` and dedicated multi-stage `Dockerfile`s for reproducible orchestration across environments.

## Consequences
- Developers on local machines can run, test, and develop the web application without requiring heavy GPU hardware by using mock inference adapters.
- Research code remains preserved and runnable on cloud GPU notebooks.
- Future model integration into production only requires implementing the `IInferenceService` interface without modifying the web layers.
