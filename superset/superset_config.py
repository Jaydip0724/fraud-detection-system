import os
import logging

# Configuring logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set a secure SECRET_KEY from an environment variable
# If the environment variable is not set, use a fixed value
SECRET_KEY = os.getenv('SECRET_KEY', 'jQx2eM7yE+Rdo8nO2fNZpzzjQYPdz+3TdOflEUX0b5B7+BaS7OgdYtfd')

# Output SECRET_KEY value to logs (for debugging only)
logger.info(f"Using SECRET_KEY: {SECRET_KEY}")

# Superset database (PostgreSQL)
SQLALCHEMY_DATABASE_URI = os.getenv('SQLALCHEMY_DATABASE_URI', 'postgresql+psycopg2://postgres:postgres@db:5432/superset')

# Superset example database (also PostgreSQL)
SQLALCHEMY_EXAMPLES_URI = os.getenv('SQLALCHEMY_EXAMPLES_URI', 'postgresql+psycopg2://postgres:postgres@db:5432/superset')
