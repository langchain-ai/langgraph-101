from langchain.chat_models import init_chat_model
import sqlite3
import requests
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

def get_engine_for_chinook_db():
    """Pull sql file, populate in-memory database, and create engine."""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

# NOTE: Configure the LLM that you want to use
from langchain.chat_models import init_chat_model

# model = init_chat_model("openai:o3-mini")
model = init_chat_model("anthropic:claude-haiku-4-5")

# Note: If you are using another `ChatModel`, you can define it in `models.py` and import it here
# from models import AZURE_OPENAI_GPT_4O
# model = AZURE_OPENAI_GPT_4O



"""Bedrock Version"""
# from dotenv import load_dotenv
# from langchain_aws import ChatBedrockConverse
# import os

# load_dotenv(dotenv_path="../.env", override=True)

# AWS_ACCESS_KEY_ID=os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY=os.getenv("AWS_SECRET_ACCESS_KEY")
# AWS_REGION_NAME=os.getenv("AWS_REGION_NAME")
# AWS_MODEL_ARN=os.getenv("AWS_MODEL_ARN")

# model = ChatBedrockConverse(
#     aws_access_key_id=AWS_ACCESS_KEY_ID,
#     aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
#     region_name=AWS_REGION_NAME,
#     provider="anthropic",
#     model_id=AWS_MODEL_ARN
# )


"""Google Vertex AI version"""
# Make sure you have your vertex ai credentials setup and your GOOGLE_APPLICATION_CREDENTIALS are pointing to the JSON file. 

# import os
# from pathlib import Path
# from dotenv import load_dotenv
# from langchain.chat_models import init_chat_model

# # Find project root and load .env
# project_root = Path().resolve().parent.parent
# load_dotenv(dotenv_path=project_root / ".env", override=True)

# # Fix credentials path to absolute
# if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
#     cred_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
#     if not os.path.isabs(cred_path):
#         os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(project_root / cred_path.lstrip("./"))

# # Create model
# model = init_chat_model("google_vertexai:gemini-2.5-flash")