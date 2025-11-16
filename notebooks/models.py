# from langchain_openai import AzureChatOpenAI
# # from langchain_anthropic import ChatAnthropic
# # from langchain_google_vertexai import ChatVertexAI
# from azure.identity import InteractiveBrowserCredential

# credential = InteractiveBrowserCredential()

# def get_token():
#     token = credential.get_token("https://cognitiveservices.azure.com/.default")
#     return token.token

# For AzureOpenAI, make sure you set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT


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

# Anthropic Haiku 4.5
#ANTHROPIC_HAIKU_4_5 = ChatAnthropic(
#    model="claude-haiku-4-5"
#)

# Vertex AI Gemini 2.5 Flash
# VERTEX_AI_GEMINI_2_5_FLASH = ChatVertexAI(
#     model="gemini-2.5-flash"
# )