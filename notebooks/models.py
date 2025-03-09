from __future__ import annotations
from typing import Optional
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings, AzureChatOpenAI
from azure.identity import InteractiveBrowserCredential
import time
import requests
from config import (AI_GATEWAY_CLIENT_ID, AI_GATEWAY_CLIENT_SECRET, AI_GATEWAY_ISSUER)

credential = InteractiveBrowserCredential()

def get_token():
    token = credential.get_token("https://cognitiveservices.azure.com/.default")
    return token.token

# For AzureOpenAI, make sure you set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT


"""
Embedding Models
"""

OPENAI_EMBEDDING_MODEL = OpenAIEmbeddings()

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


"""
Deere Gateway
"""


token_cache = {
    "token": None,
    "expires_at": 0
}

def get_ai_gateway_access_token(force_new_token: bool = False):
    current_time = time.time()
    # Check if we have a valid token in the cache
    if not force_new_token and token_cache["token"] and token_cache["expires_at"] > current_time:
        return token_cache["token"]

    token_endpoint = f"{AI_GATEWAY_ISSUER}/v1/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": AI_GATEWAY_CLIENT_ID,
        "client_secret": AI_GATEWAY_CLIENT_SECRET,
        "scope": "mlops.deere.com/model-deployments.llm.region-restricted-invocations",
    }

    response = requests.post(token_endpoint, headers=headers, data=data)
    response_json = response.json()

    if response.status_code == 200:
        # Cache the new token and its expiration time (5 minutes from now)
        token_cache["token"] = response_json["access_token"]
        token_cache["expires_at"] = current_time + \
            300  # 300 seconds = 5 minutes
        return token_cache["token"]
    else:
        raise Exception(
            f"Failed to retrieve bearer token: {response_json.get('error_description', 'Unknown error')}"
        )


class DeereAIGatewayChatOpenAI(ChatOpenAI):
    def __init__(
        self,
        access_token: str,
        base_url: str,
        model: str,
        deere_ai_gateway_registration_id: Optional[str] = None,
    ):
        base_url = (
            base_url +
            "openai/" if base_url.endswith("/") else base_url + "/openai/"
        )

        headers = {"Authorization": f"Bearer {access_token}"}
        if deere_ai_gateway_registration_id is not None:
            headers["deere-ai-gateway-registration-id"] = deere_ai_gateway_registration_id

        super(DeereAIGatewayChatOpenAI, self).__init__(
            # This is required to pass validations in the parent class
            api_key="sk-000000000000000000000000000000000000000000000000",
            model=model,
            base_url=base_url,
            default_headers=headers,
        )

from config import (AI_GATEWAY_BASE_URL, AI_GATEWAY_REGISTRATION_ID)

DEERE_CHAT_MODEL = DeereAIGatewayChatOpenAI(
        model="gpt-4o-mini-2024-07-18",
        access_token=get_ai_gateway_access_token(),
        base_url=AI_GATEWAY_BASE_URL,
        deere_ai_gateway_registration_id=AI_GATEWAY_REGISTRATION_ID
    )