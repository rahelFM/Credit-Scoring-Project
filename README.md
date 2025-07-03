## Credit Scoring Project
This project develops a credit risk prediction model for customer buy-now-pay-later services. It covers the full ML lifecycle, including data processing, model training with hyperparameter tuning, evaluation, and deployment as a REST API using FastAPI with Docker containerization and CI/CD pipeline setup.
1. Data Preparation
•	Loaded and cleaned the customer data.
•	Engineered features relevant for credit risk prediction.
•	Split data into training and testing sets.
2. Model Training and Hyperparameter Tuning
•	Trained multiple models (Logistic Regression, Random Forest) using Grid Search CV.
•	Tuned hyperparameters to improve performance.
•	Evaluated models with metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.
•	Logged models and metrics using MLflow.
3. Model Selection
•	Compared model performance and selected the best-performing model based on F1-score and ROC-AUC.
•	Saved the best model as a .pkl file.
4. API Development
•	Developed a REST API using FastAPI.
•	Created /predict endpoint to accept customer feature inputs and return risk probability.
•	Validated inputs and outputs with Pydantic models.
•	Loaded the saved model within the API for inference.
5. Containerization
•	Created a Dockerfile to containerize the FastAPI service.
•	Built a Docker image and ran the API inside a Docker container exposing port 8000.
6. Continuous Integration (CI)
•	Configured GitHub Actions workflow for:
o	Running code linter (flake8) to maintain code style.
o	Running unit tests with pytest.
•	Ensured the build pipeline fails if linting or tests fail.
 The detailed methods applied are: 
1. **Proxy Target Variable Engineering**  
   - Created a proxy target variable `is_high_risk` by clustering customers based on RFM (Recency, Frequency, Monetary) features.
   - Labeled customers as high risk or low risk based on clustering results.

2. **Feature Engineering**  
   - Processed raw customer transaction data.
   - Created and transformed features suitable for model training.

3. **Model Training and Evaluation**  
   - Trained multiple machine learning models (logistic regression and others).
   - Tuned hyperparameters to optimize performance.
   - Evaluated models using relevant metrics and selected the best model.

4. **Model Serialization**  
   - Saved the final trained logistic regression model as a pickle file for deployment.

5. **API Development**  
   - Developed a FastAPI application exposing a `/predict` endpoint.
   - The API accepts a customer ID and returns the predicted credit risk probability.

6. **Dockerization**  
   - Created a Dockerfile to containerize the API.
   - Configured volume mounts for the model and data files.
   - Successfully built and ran the API Docker container locally.

7. **CI/CD Setup with GitHub Actions**  
   - Created a GitHub Actions workflow that runs on every push to the main branch.
   - The workflow runs `flake8` for code linting.
   - Executes `pytest` for unit testing.
   - Ensures code quality and prevents broken builds.

---

## How to Use

1. Clone the repository.

2. Build and run the Docker container:
   ```bash
   docker build -t credit-scoring-api .
   docker run -d -p 8000:8000 \
     -v /path/to/models:/app/models \
     -v /path/to/data:/app/data \
     credit-scoring-api
3.	Send a POST request to predict credit risk:
4.	curl -X POST "http://localhost:8000/predict" \
5.	     -H "Content-Type: application/json" \
6.	     -d '{ "customer_id": "CustomerId_4683" }'
________________________________________
Technologies Used
•	Python 3.11
•	FastAPI
•	Scikit-learn
•	Pandas
•	Docker
•	GitHub Actions (CI/CD)
•	Flake8 (Linting)
•	Pytest (Testing)
________________________________________
Project Structure
•	src/ — Source code for the API and feature engineering
•	models/ — Serialized machine learning models
•	data/ — Input data files used for prediction and training
•	.github/workflows/ci.yml — GitHub Actions workflow file
________________________________________
Future Work
•	Expand unit test coverage.
•	Add logging and monitoring for API.
•	Deploy the containerized API to cloud services.
•	Improve feature engineering with domain-specific insights.

