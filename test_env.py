from pathlib import Path
from dotenv import load_dotenv
import os

# point to the .env in your project root
env_path = Path(__file__).parent / ".env"
loaded = load_dotenv(env_path)
print(f"Loaded .env? {loaded}")

# now fetch your vars
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
print("TIME_SCALE_SERVICE_URL:", os.getenv("TIME_SCALE_SERVICE_URL"))
