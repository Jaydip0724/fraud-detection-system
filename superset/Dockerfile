FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file from the project root directory to the container
COPY requirements.txt /app/requirements.txt

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install psycopg2-binary separately to avoid dependency problems
RUN apt-get update && apt-get install -y libpq-dev && pip install --no-cache-dir psycopg2-binary && \
    apt-get remove -y libpq-dev && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Copy the entire project into a container
COPY . /app/

# Specify the port to be used by the application
EXPOSE 8000

# Command to start the Flask application
CMD ["python", "main.py"]
