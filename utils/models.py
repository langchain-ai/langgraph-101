"""
Model Initialization File

Configures the LLM model used throughout the workshop notebook.

Default: Anthropic (claude-haiku-4-5).

To swap providers:
  1. Comment out the Default Models section below.
  2. Uncomment the section for your desired provider.
  3. Follow the setup notes inline.

Provider sections included (commented out by default):
  - Azure OpenAI  (needs AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT)
  - AWS Bedrock   (needs AWS credentials + AWS_MODEL_ARN)
  - Google Vertex (needs GOOGLE_APPLICATION_CREDENTIALS)
"""

from dotenv import load_dotenv
load_dotenv(override=True)
from langchain.chat_models import init_chat_model


# ---- Default Models -------------------------------------------------------
# model = init_chat_model("openai:gpt-4.1-mini")

# Use Anthropic by default
model = init_chat_model("anthropic:claude-haiku-4-5")


# ---- Azure OpenAI ---------------------------------------------------------
# from langchain_openai import AzureChatOpenAI
# from azure.identity import InteractiveBrowserCredential

# credential = InteractiveBrowserCredential()

# def get_token():
#     token = credential.get_token("https://cognitiveservices.azure.com/.default")
#     return token.token

# Make sure AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are set.

# Azure OpenAI: Using environment variables
# model = AzureChatOpenAI(
#     azure_deployment="gpt-4o",
#     streaming=True,
# )

# Azure OpenAI: Using Azure AD
# model = AzureChatOpenAI(
#     api_version="2024-03-01-preview",
#     azure_endpoint="https://deployment.openai.azure.com/",
#     azure_deployment="gpt-4o",
#     azure_ad_token_provider=get_token,
# )


# ---- AWS Bedrock ----------------------------------------------------------
# import os
# from langchain_aws import ChatBedrockConverse

# AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
# AWS_MODEL_ARN = os.getenv("AWS_MODEL_ARN")

# model = ChatBedrockConverse(
#     aws_access_key_id=AWS_ACCESS_KEY_ID,
#     aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#     region_name=AWS_REGION_NAME,
#     provider="anthropic",
#     model_id=AWS_MODEL_ARN,
# )


# ---- Google Vertex AI -----------------------------------------------------
# Make sure your Vertex AI credentials are set up and GOOGLE_APPLICATION_CREDENTIALS
# points to the JSON file.

# import os
# from pathlib import Path
# from langchain.chat_models import init_chat_model

# # Resolve project root and load .env (utils/ -> project root is one level up)
# project_root = Path(__file__).resolve().parent.parent
# load_dotenv(dotenv_path=project_root / ".env", override=True)

# # Make the credentials path absolute if it was given as a relative path
# if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
#     cred_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
#     if not os.path.isabs(cred_path):
#         os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(project_root / cred_path.lstrip("./"))

# model = init_chat_model("google_vertexai:gemini-2.5-flash")
