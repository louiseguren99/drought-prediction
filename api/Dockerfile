# Python 3.13 matches the environment the model was trained in (scikit-learn 1.9.1)
FROM python:3.13-slim

WORKDIR /app

# Install the exact library versions first (this layer is cached between builds)
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only the application code goes in the image, the model is mounted at run time
COPY api/app.py .

# The model file is NOT in the image (models/ is in .gitignore). Mount it, see README.
ENV MODEL_PATH=/models/best_model_h1.joblib

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
