# src/chatbot_ui/config.py

import os
from dotenv import load_dotenv

# Load environment variables from the specified .env file
load_dotenv()
print(f"DEBUG: Loaded environment variables")


# Define configuration settings
class Config:
    DEFAULT_SYSTEM_PROMPT = os.getenv("DEFAULT_SYSTEM_PROMPT", "")
    MAX_MAX_NEW_TOKENS = int(os.getenv("MAX_MAX_NEW_TOKENS", 2048))
    DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", 1024))
    MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", 4000))
    MODEL_PATH = os.getenv("MODEL_PATH")
    assert MODEL_PATH is not None, f"MODEL_PATH is required, got: {MODEL_PATH}"
    BACKEND_TYPE = os.getenv("BACKEND_TYPE")
    assert BACKEND_TYPE is not None, f"BACKEND_TYPE is required, got: {BACKEND_TYPE}"
    LOAD_IN_8BIT = os.getenv("LOAD_IN_8BIT", "False").lower() == "true"


# Get the absolute path of the project directory
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Set the GRADIO_CACHE_DIR environment variable
cache_dir = os.path.join(os.path.dirname(__file__), "chatbot_ui", "gradio_cached_examples")
os.environ['GRADIO_CACHE_DIR'] = cache_dir
