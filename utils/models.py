"""
Model Initialization File

This file configures the LLM model to be used throughout the application.

Default Configuration:
- The default provider is OpenAI (o3-mini model)
- You can also switch to Anthropic by uncommenting the alternative model line

Alternative Providers:
To use a different LLM provider, follow these steps:
1. Comment out the "Default Models" section below
2. Uncomment the section for your desired provider:
   - Azure OpenAI: Requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT
   - AWS Bedrock: Requires AWS credentials and configuration
   - Google Vertex AI: Requires GOOGLE_APPLICATION_CREDENTIALS setup
3. Follow the setup instructions within each section
"""

"""Default Models"""
from dotenv import load_dotenv
load_dotenv(dotenv_path="../../.env", override=True)
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:o3-mini")

# Use Anthropic instead of OpenAI
# model = init_chat_model("anthropic:claude-haiku-4-5")


"""AZURE OpenAI Version"""
# from langchain_openai import AzureChatOpenAI
# # from langchain_anthropic import ChatAnthropic
# # from langchain_google_vertexai import ChatVertexAI
# from azure.identity import InteractiveBrowserCredential

# credential = InteractiveBrowserCredential()

# def get_token():
#     token = credential.get_token("https://cognitiveservices.azure.com/.default")
#     return token.token

# For AzureOpenAI, make sure you set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT

# Azure OpenAI: Using Environment Variables
# AZURE_OPENAI_GPT_4O = AzureChatOpenAI(
#     azure_deployment="gpt-4o",
#     streaming=True
# )

# Azure OpenAI: Using Azure AD
# AZURE_OPENAI_GPT_4O = AzureChatOpenAI(
#     api_version="2024-03-01-preview",
#     azure_endpoint="https://deployment.openai.azure.com/",
#     azure_deployment="gpt-4o",
#     azure_ad_token_provider=get_token
# )


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
# # Use __file__ to get the location of this file, then go up two directories to project root
# project_root = Path(__file__).resolve().parent.parent
# load_dotenv(dotenv_path=project_root / ".env", override=True)

# # Fix credentials path to absolute
# if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
#     cred_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
#     if not os.path.isabs(cred_path):
#         os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(project_root / cred_path.lstrip("./"))

# # Create model
# model = init_chat_model("google_vertexai:gemini-2.5-flash")