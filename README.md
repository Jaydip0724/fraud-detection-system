# Fraud Detection System

A containerized fraud detection system that leverages machine learning to identify fraudulent credit card transactions. This project demonstrates a complete data science pipeline—from data processing and model training to real-time API monitoring and BI visualization using Apache Superset.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Architecture](#architecture)
- [Testing](#testing)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

## Overview

This project is a comprehensive solution for fraud detection in credit card transactions. It includes:

- **Data Processing:** Loading, cleaning, and preprocessing data.
- **Model Training:** Using a RandomForest classifier for fraud detection.
- **Streaming Simulation:** Generating synthetic transaction data and simulating a live data stream.
- **API Endpoints:** Providing endpoints to monitor system status, metrics, statistics, and transactions.
- **Visualization:** Leveraging Apache Superset for interactive dashboarding and data exploration.
- **Containerization:** Packaging all services using Docker and orchestrating them with Docker Compose.

## Features

- **Machine Learning Model:** Trained with historical data to detect fraudulent transactions.
- **Real-time Data Processing:** Simulates a continuous stream of transactions with on-the-fly model updates.
- **REST API:** Exposes endpoints for tracking model metrics and transaction logs.
- **BI Dashboard:** Integrated Apache Superset instance for visual exploration of transaction data.
- **Modular Codebase:** Structured into multiple modules for ease of testing and scaling.
- **Containerized Deployment:** Easily deployable with Docker and Docker Compose.

## Repository Structure

```
fraud-detection-system/
├── app/
│   ├── main.py                  # Main application code (Flask server)
│   ├── utils/                   # Utility modules
│   │   ├── data_processing.py   # Data preprocessing functions
│   │   ├── model.py             # Machine learning model functions
│   │   └── metrics.py           # Model metric calculation and logging
├── data/
│   └── test_creditcard_2023.csv # Demo dataset for testing
├── docs/
│   ├── model_explanation.md     # Explanation of the ML model
│   ├── api_reference.md         # API documentation
│   └── system_architecture.md   # System architecture overview
├── notebooks/
│   ├── exploratory_analysis.ipynb # Exploratory data analysis of transactions
│   └── model_development.ipynb  # Model development and testing
├── superset/
│   ├── Dockerfile               # Dockerfile for Superset service
│   └── superset_config.py       # Superset configuration
├── tests/                       # Unit tests for the project
├── .env.example                 # Template for environment variables
├── .gitignore
├── docker-compose.yml           # Docker Compose configuration for all services
├── Dockerfile                   # Dockerfile for the Flask application
├── LICENSE                      # Project license (e.g., MIT)
├── README.md                    # Project README (this file)
└── requirements.txt             # Python dependencies
```

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/fraud-detection-system.git
   cd fraud-detection-system
   ```

2. **Configure Environment Variables:**
   - Copy the environment variables template:
     ```bash
     cp .env.example .env
     ```
   - Modify the `.env` file if necessary.

3. **Build and Run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```
   This command starts the following services:
   - **db:** PostgreSQL database.
   - **web:** Flask application running on port 8000.
   - **superset:** Apache Superset dashboard running on port 8088.

## Usage

- **Flask API:**
  - Check the app status at: [http://localhost:8000/](http://localhost:8000/)
  - The app simulates real-time data ingestion and updates model metrics.

- **Apache Superset:**
  - Access the Superset dashboard at: [http://localhost:8088/](http://localhost:8088/)
  - Use Superset to build interactive dashboards and explore transaction data.

## API Endpoints

- **GET /**  
  Returns the application status and current server time.
  ```json
  {
    "status": "running",
    "time": "2023-10-XXTXX:XX:XX"
  }
  ```

- **GET /metrics**  
  Retrieves the latest model performance metrics (precision, recall, F1 score, and accuracy).

- **GET /statistics**  
  Provides detailed statistics (mean and standard deviation) for each feature.

- **GET /transactions**  
  Lists the latest 100 transactions, including actual and predicted classes.

For more details, refer to [docs/api_reference.md](docs/api_reference.md).

## Architecture

The system is composed of several modular components:
- **Data Processing & Model Training:**  
  Located in `app/utils/`, these modules handle data cleaning, feature scaling, model training, and predictions.
  
- **Flask Application:**  
  Acts as the API gateway for handling data streams, processing transactions, and logging to the database.
  
- **Database:**  
  A PostgreSQL database stores transaction logs, model metrics, and data statistics.
  
- **BI Dashboard:**  
  Apache Superset (configured in the `superset/` directory) provides interactive data visualization capabilities.
  
- **Containerization:**  
  Docker and Docker Compose ensure environment consistency and ease of deployment.

For additional details, see [docs/system_architecture.md](docs/system_architecture.md).

## Testing

Unit tests are provided in the `tests/` directory. To run the tests:
```bash
# Install dependencies if not already installed
pip install -r requirements.txt

# Run all tests using unittest
python -m unittest discover -s tests
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact:  
**Uladzimir Manulenka**  
[ vlma@tut.by ](mailto:vlma@tut.by)
