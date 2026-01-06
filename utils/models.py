"""
Model Initialization File

This file configures the LLM model to be used throughout the application.
"""

"""Default Models"""
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env", override=True)
from aipe.llm import init_payx_chat_model

model = init_payx_chat_model("gpt-4o", model_provider="azure_openai")
