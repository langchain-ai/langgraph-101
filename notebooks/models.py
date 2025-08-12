from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI
from azure.identity import InteractiveBrowserCredential

credential = InteractiveBrowserCredential()

def get_token():
    token = credential.get_token("https://cognitiveservices.azure.com/.default")
    return token.token

# For AzureOpenAI, make sure you set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT


"""
Embedding Models
"""

EMBEDDING_MODEL = VertexAIEmbeddings(model_name="text-embedding-large-exp-03-07")

# Azure OpenAI: Using Environment Variables
# AZURE_OPENAI_EMBEDDING_MODEL = AzureOpenAIEmbeddings(
#     model="text-embedding-3-large",
# )

# Azure OpenAI: Using Azure AD
# AZURE_OPENAI_EMBEDDING_MODEL = AzureOpenAIEmbeddings(
#     openai_api_version="2024-03-01-preview",
#     azure_endpoint="https://deployment.openai.azure.com/",
#     model="text-embedding-3-large",
#     azure_ad_token_provider=get_token
# )


"""
Chat Models
"""
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

# Anthropic Sonnet 3.5 
# ANTHROPIC_SONNET_3_5 = ChatAnthropic(
#     model="claude-3-5-sonnet-20240620",
#     streaming=True
# )

# Vertex AI Gemini 1.5 Flash
# VERTEX_AI_GEMINI_1_5_FLASH = ChatVertexAI(
#     model="gemini-1.5-flash-002",
#     streaming=True
# )