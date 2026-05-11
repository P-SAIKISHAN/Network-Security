### Network Security Projects for Phising Data 
# 🔐 Network Security - Phishing Detection System

<div align="center">

![Network Security](https://img.shields.io/badge/Network-Security-red?style=for-the-badge)
![ML Pipeline](https://img.shields.io/badge/ML-Pipeline-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.7+-green?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-yellow?style=for-the-badge)
![MongoDB](https://img.shields.io/badge/MongoDB-4.0+-green?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Supported-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A comprehensive machine learning-based network security solution for detecting phishing attacks**

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [API](#-api-endpoints) • [Contributing](#-contributing)

</div>

---

## 📋 Overview

This is a production-grade machine learning project designed to detect and classify phishing attacks in network traffic. It implements a complete end-to-end ML pipeline with data processing, model training, evaluation, and a REST API for real-time predictions.

The project follows industry best practices including:
- ✅ Modular architecture with clean code
- ✅ Comprehensive error handling and logging
- ✅ MongoDB integration for data persistence
- ✅ MLflow integration for experiment tracking
- ✅ Docker containerization for deployment
- ✅ FastAPI for REST API endpoints
- ✅ DVC for data and model versioning

---

## 🎯 Project Objectives

- 🔍 Identify and classify phishing data in network traffic
- 🤖 Build robust machine learning models for threat detection
- 📊 Process and validate large network security datasets
- 🌐 Provide actionable insights for network defense mechanisms
- 📈 Track experiments and model performance metrics
- 🚀 Enable easy deployment and scaling

---

## 🏗️ System Architecture

### Overall System Architecture

The system consists of multiple interconnected layers:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NETWORK SECURITY SYSTEM                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐             │
│  │  Data        │   │  Data        │   │  Data        │             │
│  │  Ingestion   │──▶│  Validation  │──▶│  Transform   │             │
│  │              │   │              │   │              │             │
│  └──────────────┘   └──────────────┘   └──────────────┘             │
│         │                  │                    │                    │
│         └──────────────────┴────────────────────┘                    │
│                            │                                         │
│                            ▼                                         │
│  ┌─────────────────────────────────┐   ┌──────────────┐            │
│  │    Model Training Pipeline       │──▶│  Trained     │            │
│  │  - Classifier Selection          │   │  Model      │            │
│  │  - Hyperparameter Tuning         │   │  (Artifacts)│            │
│  │  - Cross-Validation              │   └──────────────┘            │
│  └─────────────────────────────────┘         │                     │
│                            │                  │                     │
│                            └──────────┬───────┘                     │
│                                       ▼                             │
│                  ┌────────────────────────────────┐                │
│                  │      FastAPI REST Server       │                │
│                  │  - /predict endpoint           │                │
│                  │  - /train endpoint             │                │
│                  │  - /health endpoint            │                │
│                  │  - Interactive API Docs        │                │
│                  └────────────────────────────────┘                │
│                                       │                             │
│         ┌─────────────────────────────┴──────────────────┐         │
│         │                                                 │         │
│         ▼                                                 ▼         │
│   ┌─────────────┐                               ┌──────────────┐  │
│   │  MongoDB    │                               │  Predictions │  │
│   │  Database   │                               │  & Metrics   │  │
│   └─────────────┘                               └──────────────┘  │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Architecture Layers

**1. Data Ingestion Layer**
- Load raw network data from CSV/JSON files
- Perform initial data exploration
- Split data into training and testing sets
- Generate data ingestion artifacts

**2. Processing Layer**
- Data validation against schema
- Quality checks and outlier detection
- Feature engineering and transformation
- Scaling and normalization

**3. ML Pipeline Layer**
- Train multiple classification models
- Hyperparameter optimization
- Cross-validation and evaluation
- Model selection based on performance

**4. API & Deployment Layer**
- FastAPI REST server
- Real-time prediction endpoints
- Health check monitoring
- Interactive API documentation

**5. Data Persistence Layer**
- MongoDB for artifact storage
- Training logs and metadata
- Predictions and results
- Model versioning

---

## 📊 ML Pipeline Workflow

### Complete 4-Stage Pipeline

```
STAGE 1: DATA INGESTION
├─ Load Raw Network Data
│  ├─ CSV Files
│  ├─ JSON Files
│  └─ Network Logs
├─ Split Data (70% Train / 30% Test)
├─ Shuffle & Randomize
└─ Generate Artifacts
         │
         ▼
STAGE 2: DATA VALIDATION
├─ Schema Validation
│  ├─ Check Data Types
│  └─ Verify Columns
├─ Quality Checks
│  ├─ Missing Values Detection
│  ├─ Outlier Detection
│  └─ Data Distribution Analysis
└─ Validation Report
         │
         ▼
STAGE 3: DATA TRANSFORMATION
├─ Feature Engineering
│  ├─ Create New Features
│  ├─ Feature Selection
│  └─ Dimension Reduction
├─ Scaling & Normalization
│  ├─ StandardScaler
│  └─ MinMaxScaler
├─ Encoding
│  ├─ One-Hot Encoding
│  └─ Label Encoding
└─ Transformation Artifacts
         │
         ▼
STAGE 4: MODEL TRAINING
├─ Train Multiple Models
│  ├─ Logistic Regression
│  ├─ Random Forest
│  ├─ SVM
│  └─ Gradient Boosting
├─ Hyperparameter Tuning
├─ Cross-Validation (k-fold)
├─ Performance Evaluation
│  ├─ Accuracy
│  ├─ Precision & Recall
│  ├─ F1-Score
│  └─ ROC-AUC
└─ ✅ Deploy Best Model
```

---

## 📁 Project Directory Structure

```
Network-Security/
│
├── 📁 networksecurity/                          # Main package
│   │
│   ├── 📁 components/                           # ML Pipeline Components
│   │   ├── data_ingestion.py                   # Load & process data
│   │   ├── data_validation.py                  # Validate data quality
│   │   ├── data_transformation.py              # Feature engineering
│   │   └── model_trainer.py                    # Train ML models
│   │
│   ├── 📁 entity/                               # Configuration Entities
│   │   └── config_entity.py                    # Config classes
│   │       ├── TrainingPipelineConfig
│   │       ├── DataIngestionConfig
│   │       ├── DataValidationConfig
│   │       ├── DataTransformationConfig
│   │       └── ModelTrainerConfig
│   │
│   ├── 📁 exception/                            # Exception Handling
│   │   └── exception.py                        # NetworkSecurityException
│   │
│   └── 📁 logging/                              # Logging Module
│       └── logger.py                           # Event logging utilities
│
├── 📁 Network_Data/                             # Raw Datasets
│   └── *.csv, *.json                           # Input data files
│
├── 📁 data_schema/                              # Data Schemas
│   └── schema definitions for validation
│
├── 📁 valid_data/                               # Validated Datasets
│   └── Processed & validated data
│
├── 📁 final_model/                              # Trained Models
│   └── Serialized model artifacts
│
├── 📁 prediction_output/                        # Predictions
│   └── Results from inference
│
├── 📁 templates/                                # Web Templates
│   └── HTML templates (optional UI)
│
├── 📁 __pycache__/                              # Python cache
│
├── ⚙️ main.py                                    # Pipeline Orchestration
│   └── Executes complete ML pipeline
│
├── ⚙️ app.py                                     # FastAPI Application
│   └── REST API server
│
├── 📦 requirements.txt                          # Dependencies
│
├── ⚙️ setup.py                                   # Package Setup
│
├── 🐳 Dockerfile                                # Docker Configuration
│
├── .dvcignore                                   # DVC ignore patterns
│
├── .gitignore                                   # Git ignore patterns
│
├── test_mongodb.py                              # MongoDB tests
│
├── check_mongo.py                               # MongoDB verification
│
└── 📄 README.md                                 # This file
```

---

## 🔧 Technology Stack

### Core Framework & Libraries
| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.7+ | Programming language |
| **FastAPI** | ≥0.68.0 | REST API framework |
| **Uvicorn** | ≥0.15.0 | ASGI server |
| **scikit-learn** | ≥0.24.0 | Machine learning |
| **Pandas** | ≥1.3.0 | Data manipulation |
| **NumPy** | ≥1.21.0 | Numerical computing |

### Database & Storage
| Technology | Version | Purpose |
|-----------|---------|---------|
| **MongoDB** | ≥4.0 | NoSQL database |
| **PyMongo** | ≥4.0 | MongoDB driver |
| **DVC** | Latest | Data versioning |

### ML & Experimentation
| Technology | Version | Purpose |
|-----------|---------|---------|
| **MLflow** | ≥1.20.0 | Experiment tracking |
| **DagsHub** | ≥0.1.34 | ML collaboration |
| **dill** | ≥0.3.4 | Serialization |

### Deployment & DevOps
| Technology | Version | Purpose |
|-----------|---------|---------|
| **Docker** | Latest | Containerization |
| **python-dotenv** | Latest | Environment variables |
| **certifi** | ≥2021.10.8 | SSL certificates |
| **dnspython** | ≥2.1.0 | DNS utilities |

---

## 📦 Complete Dependencies

```txt
python-dotenv>=0.19.0          # Environment configuration
pandas>=1.3.0                  # Data manipulation
numpy>=1.21.0                  # Numerical computing
pymongo>=4.0                   # MongoDB driver
certifi>=2021.10.8             # SSL certificates
dnspython>=2.1.0               # DNS utilities
dill>=0.3.4                    # Object serialization
mlflow>=1.20.0                 # ML lifecycle management
dagshub>=0.1.34                # ML experiment tracking
pyaml>=20.4.0                  # YAML support
fastapi>=0.68.0                # REST API framework
uvicorn>=0.15.0                # ASGI server
scikit-learn>=0.24.0           # Machine learning library
python-multipart>=0.0.5        # Multipart form support
```

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.7 or higher
- **MongoDB**: Running instance (local or cloud)
- **Git**: For version control
- **pip/conda**: Package manager

### Installation Steps

#### 1️⃣ Clone Repository
```bash
git clone https://github.com/P-SAIKISHAN/Network-Security.git
cd Network-Security
```

#### 2️⃣ Create Virtual Environment
```bash
# Using venv
python -m venv venv

# Activate
# On Linux/Mac
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

#### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4️⃣ Configure Environment
Create `.env` file in root directory:
```env
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/
DATABASE_NAME=network_security_db
COLLECTION_NAME=phishing_data

# Application Settings
LOG_LEVEL=INFO
DEBUG=False

# Model Settings
MODEL_PATH=final_model/
PREDICTION_OUTPUT_PATH=prediction_output/
```

#### 5️⃣ Verify MongoDB Connection
```bash
python check_mongo.py
python test_mongodb.py
```

### Running the Application

#### 🔄 Run ML Pipeline
```bash
python main.py
```

**Output:**
```
[INFO] Initiate the data ingestion
[INFO] Data initiation completed
[INFO] Initiate the data Validation
[INFO] Data validation completed
[INFO] data Transformation started
[INFO] data Transformation completed
[INFO] Model Training started
[INFO] Model Training completed
[INFO] ✅ Pipeline execution successful
```

#### 🌐 Start API Server
```bash
python app.py
```

**Server starts at:** `http://localhost:8000`

#### 📚 Access Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

---

## 🌐 API Endpoints

### 1. **Predict Endpoint** - `POST /predict`

Make predictions on new data.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [1.2, 3.4, 5.6, 2.1, 4.3, 3.2, 5.1, 2.8],
    "metadata": {"source": "network_log"}
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "confidence": 0.92,
  "threat_level": "HIGH",
  "explanation": "Patterns detected similar to known phishing attempts",
  "timestamp": "2026-02-26T10:30:45.123Z"
}
```

### 2. **Training Endpoint** - `POST /train`

Trigger the complete training pipeline.

**Request:**
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "Network_Data/",
    "model_name": "phishing_detector_v1",
    "test_size": 0.3
  }'
```

**Response:**
```json
{
  "status": "success",
  "message": "Training pipeline completed successfully",
  "model_name": "phishing_detector_v1",
  "metrics": {
    "accuracy": 0.945,
    "precision": 0.938,
    "recall": 0.952,
    "f1_score": 0.945
  },
  "training_time": "45.23 seconds"
}
```

### 3. **Health Check** - `GET /health`

Check API health status.

**Request:**
```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "database_connected": true,
  "timestamp": "2026-02-26T10:30:45.123Z"
}
```

### 4. **API Documentation** - `GET /docs`

Interactive Swagger UI documentation.

### 5. **Alternative Documentation** - `GET /redoc`

ReDoc alternative documentation.

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t network-security:latest .
```

### Run Container

```bash
docker run -p 8000:8000 \
  -e MONGODB_URI=mongodb://host.docker.internal:27017/ \
  -e DATABASE_NAME=network_security_db \
  network-security:latest
```

### Docker Compose (Optional)

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  mongodb:
    image: mongo:latest
    container_name: network_security_db
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: password
    volumes:
      - mongo_data:/data/db

  app:
    build: .
    container_name: network_security_api
    ports:
      - "8000:8000"
    environment:
      MONGODB_URI: mongodb://admin:password@mongodb:27017/
      DATABASE_NAME: network_security_db
    depends_on:
      - mongodb
    volumes:
      - ./Network_Data:/app/Network_Data
      - ./final_model:/app/final_model

volumes:
  mongo_data:
```

Run with:
```bash
docker-compose up -d
```

---

## 📊 Data Flow Architecture

### End-to-End Data Flow

```
┌──────────────┐
│  Raw Data    │
│  Sources     │
│  (CSV/JSON)  │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│  Data Ingestion      │
│  - Load & Parse      │
│  - Train/Test Split  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Data Validation     │
│  - Schema Check      │
│  - Quality Metrics   │
└──────┬───────────────┘
       │
       ├─────────────────────┐
       │                     │
       ▼                     ▼
┌──────────────────┐  ┌─────────────┐
│ Valid Data       │  │  MongoDB    │
│ - Artifacts      │  │  - Metadata │
│ - Logs           │  │  - Status   │
└──────┬───────────┘  └─────────────┘
       │
       ▼
┌──────────────────────────────┐
│  Data Transformation         │
│  - Feature Engineering       │
│  - Scaling & Normalization   │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  Model Training              │
│  - Classifier Training       │
│  - Hyperparameter Tuning     │
│  - Cross-Validation          │
└──────┬───────────────────────┘
       │
       ├─────────────────────┐
       │                     │
       ▼                     ▼
┌──────────────────┐  ┌─────────────────┐
│ Final Model      │  │  MLflow/DVC     │
│ - Serialized     │  │  - Experiment   │
│ - Artifacts      │  │  - Tracking     │
└──────┬───────────┘  └─────────────────┘
       │
       ▼
┌──────────────────────┐
│  FastAPI Server      │
│  - REST Endpoints    │
│  - Predictions       │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Output              │
│  - Predictions       │
│  - Metrics           │
│  - Logs              │
└──────────────────────┘
```

---

## ✨ Key Features

### 🔍 Comprehensive Data Processing
- ✅ Multi-format data support (CSV, JSON)
- ✅ Automatic schema validation
- ✅ Missing value detection and handling
- ✅ Outlier identification and treatment
- ✅ Advanced feature engineering

### 🤖 Robust ML Pipeline
- ✅ Multiple classifier support
- ✅ Automatic hyperparameter tuning
- ✅ Cross-validation (k-fold)
- ✅ Model comparison and selection
- ✅ Performance metrics tracking

### 🔐 Production-Ready
- ✅ Error handling and recovery
- ✅ Comprehensive logging
- ✅ Database integration
- ✅ Docker containerization
- ✅ Environment configuration

### 🌐 REST API
- ✅ FastAPI framework
- ✅ Interactive API documentation
- ✅ Real-time predictions
- ✅ Health monitoring
- ✅ Multipart file support

### 📈 Experiment Tracking
- ✅ MLflow integration
- ✅ DagsHub collaboration
- ✅ DVC data versioning
- ✅ Metric logging
- ✅ Parameter management

### 📊 Data Persistence
- ✅ MongoDB integration
- ✅ Artifact storage
- ✅ Log management
- ✅ Metadata tracking
- ✅ Query capabilities

---

## 🔧 Configuration Management

### Configuration Classes

```python
# TrainingPipelineConfig
- artifact_dir: str
- timestamp: str

# DataIngestionConfig
- data_ingestion_dir: str
- raw_data_path: str
- ingested_train_dir: str
- ingested_test_dir: str

# DataValidationConfig
- data_validation_dir: str
- valid_train_dir: str
- valid_test_dir: str
- invalid_train_dir: str
- invalid_test_dir: str

# DataTransformationConfig
- data_transformation_dir: str
- transformed_train_dir: str
- transformed_test_dir: str
- transformed_object_file_path: str

# ModelTrainerConfig
- model_trainer_dir: str
- trained_model_file_path: str
```

### Environment Variables

```bash
# Database
MONGODB_URI=mongodb://localhost:27017/
DATABASE_NAME=network_security_db
COLLECTION_NAME=phishing_data

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Model
MODEL_PATH=final_model/
PREDICTION_OUTPUT_PATH=prediction_output/

# API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False
```

---

## 🔐 Error Handling

### Exception Hierarchy

```
NetworkSecurityException
├── DataIngestionException
├── DataValidationException
├── DataTransformationException
├── ModelTrainerException
└── PredictionException
```

### Exception Features
- ✅ Custom error messages
- ✅ Stack trace preservation
- ✅ Automatic logging
- ✅ Graceful degradation
- ✅ Recovery mechanisms

---

## 📈 Monitoring & Tracking

### MLflow Integration
```python
mlflow.set_experiment("network_security")
mlflow.log_params(model_params)
mlflow.log_metrics(metrics)
mlflow.log_artifact(model_path)
mlflow.sklearn.log_model(model, "model")
```

### DVC Integration
```bash
dvc add Network_Data/
dvc add final_model/
dvc push
```

### Logging
```python
from networksecurity.logging.logger import logging

logging.info("Starting pipeline")
logging.error("Error occurred")
logging.warning("Warning message")
```

---

## 🧪 Testing

### MongoDB Tests

```bash
# Test connection
python test_mongodb.py

# Check configuration
python check_mongo.py
```

### Data Pipeline Tests

```bash
# Test data ingestion
python -c "from networksecurity.components.data_ingestion import DataIngestion"

# Validate imports
python -m pytest
```

---

## 📚 Additional Scripts

### Data Utilities

**push_data.py** - Upload datasets to MongoDB
```bash
python push_data.py --data_path Network_Data/ --collection phishing_data
```

**check_mongo.py** - Verify MongoDB connectivity
```bash
python check_mongo.py
```

**test_mongodb.py** - Run MongoDB tests
```bash
python test_mongodb.py
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

### 1. Fork the Repository
```bash
git clone https://github.com/yourusername/Network-Security.git
cd Network-Security
```

### 2. Create Feature Branch
```bash
git checkout -b feature/AmazingFeature
```

### 3. Make Changes
- Write clean, documented code
- Follow PEP 8 style guide
- Add tests for new features

### 4. Commit Changes
```bash
git commit -m "Add: Amazing feature description"
```

### 5. Push to Branch
```bash
git push origin feature/AmazingFeature
```

### 6. Open Pull Request
- Describe your changes
- Reference related issues
- Await review and feedback

### Contribution Guidelines
- ✅ Follow existing code style
- ✅ Write comprehensive docstrings
- ✅ Add unit tests
- ✅ Update documentation
- ✅ Keep commits atomic

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Sai Kishan (P-SAIKISHAN)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

---

## 📝 Changelog

### Version 1.0.0 (2026-02-26)
- Initial release
- Core ML pipeline implementation
- FastAPI integration
- MongoDB support
- Comprehensive documentation
- Docker containerization

---

<div align="center">

### Made with ❤️ by [Sai Kishan](https://github.com/P-SAIKISHAN)

**[⬆ Back to Top](#-network-security---phishing-detection-system)**

---

![GitHub Stars](https://img.shields.io/github/stars/P-SAIKISHAN/Network-Security?style=social)
![GitHub Forks](https://img.shields.io/github/forks/P-SAIKISHAN/Network-Security?style=social)
![GitHub Issues](https://img.shields.io/github/issues/P-SAIKISHAN/Network-Security?style=social)

**Last Updated**: February 26, 2026  
**Status**: ✅ Active Development  
**License**: MIT

</div>