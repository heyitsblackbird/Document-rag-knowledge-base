from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()
# Access the GEMINI API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")