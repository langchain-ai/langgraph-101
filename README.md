# LangGraph 101 - Google Vertex AI Branch

Welcome to LangGraph 101 configured for Google Vertex AI! 

## Introduction
This branch of LangGraph 101 is specifically configured for use with Google Vertex AI. You will learn about the fundamentals of LangGraph through notebooks that use Vertex AI models. This is a condensed version of LangChain Academy, and is intended to be run in a session with a LangChain engineer. If you're interested in going into more depth, or working through a tutorial on your own, check out LangChain Academy [here](https://academy.langchain.com/courses/intro-to-langgraph)!

## Context

At LangChain, we aim to make it easy to build LLM applications. One type of LLM application you can build is an agent. There's a lot of excitement around building agents because they can automate a wide range of tasks that were previously impossible. 

In practice though, it is incredibly difficult to build systems that reliably execute on these tasks. As we've worked with our users to put agents into production, we've learned that more control is often necessary. You might need an agent to always call a specific tool first or use different prompts based on its state.

To tackle this problem, we've built [LangGraph](https://langchain-ai.github.io/langgraph/) — a framework for building agent and multi-agent applications. Separate from the LangChain package, LangGraph's core design philosophy is to help developers add better precision and control into agent workflows, suitable for the complexity of real-world systems.

## Google Vertex AI Setup

### 1. Clone the Repository
```bash
git clone https://github.com/langchain-ai/langgraph-101.git
cd langgraph-101
git checkout vertex
```

### 2. Download Vertex AI Credentials
1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to IAM & Admin → Service Accounts
3. Create a new service account or select an existing one with Vertex AI permissions
4. Click "Keys" → "Add Key" → "Create New Key" → "JSON"
5. Download the JSON credentials file

### 3. Configure Environment Variables
1. Create a `.env` file in the project root
2. Add the path to your Vertex AI credentials file:
```
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/vertex-credentials.json
```

### 4. Install Dependencies
Follow the setup instructions in `setup.md` for installing the Python environment and required packages.

**Note:** This branch is exclusively configured for Google Vertex AI. For other providers (OpenAI, Azure OpenAI), please use the main branch.