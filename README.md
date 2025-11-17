# LangGraph 101

Welcome to LangGraph 101! 

## Introduction
This repository contains hands-on tutorials for learning LangChain and LangGraph, organized into two learning tracks:

- **LG101**: Fundamentals of building agents with LangChain v1 and LangGraph v1
- **LG201**: Advanced patterns including multi-agent systems and production workflows

This is a condensed version of LangChain Academy, intended to be run in a session with a LangChain engineer. If you're interested in going into more depth, or working through tutorials on your own, check out [LangChain Academy](https://academy.langchain.com/courses/intro-to-langgraph)! LangChain Academy has helpful pre-recorded videos from our LangChain engineers.

## What's Inside

### LG101 - Fundamentals
- **langgraph_101.ipynb**: Build your first agent with models, tools, memory, and streaming
- **langgraph_102.ipynb**: Advanced concepts including middleware and human-in-the-loop patterns

### LG201 - Production Patterns  
- **email_agent.ipynb**: Build a stateful email triage and response agent
- **multi_agent.ipynb**: Multi-agent systems with supervisors and specialized sub-agents

All notebooks use the latest **LangChain v1** and **LangGraph v1** primitives, including `create_agent()`, middleware, and the new interrupt patterns.

## Context

At LangChain, we aim to make it easy to build LLM applications. One type of LLM application you can build is an agent. There's a lot of excitement around building agents because they can automate a wide range of tasks that were previously impossible. 

In practice though, it is incredibly difficult to build systems that reliably execute on these tasks. As we've worked with our users to put agents into production, we've learned that more control is often necessary. You might need an agent to always call a specific tool first or use different prompts based on its state.

To tackle this problem, we've built [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — a framework for building agent and multi-agent applications. Separate from the LangChain package, LangGraph's core design philosophy is to help developers add better precision and control into agent workflows, suitable for the complexity of real-world systems.

## Pre-work

### Clone the LangGraph 101 repo
```
git clone https://github.com/langchain-ai/langgraph-101.git
```


### Create an environment 
Ensure you have a recent version of pip and python installed
```
$ cd langgraph-101
# Copy the .env.example file to .env
cp .env.example .env
```
If you run into issues with setting up the python environment or acquiring the necessary API keys due to any restrictions (ex. corporate policy), contact your LangChain representative and we'll find a work-around!

### Package Installation
Ensure you have a recent version of pip and python installed
```
# Install uv if you haven't already
pip install uv

# Install the package, allowing for pre-release 
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

### Running Agents Locally

You can run the agents in this repository locally using `langgraph dev`. This gives you:
- A local API server for your agents
- LangGraph Studio UI for testing and debugging
- Hot-reloading during development

```bash
# From the root directory, start the LangGraph development server
langgraph dev

# This will start a local server and provide:
# - API endpoint for your agents (typically http://localhost:8123)
# - LangGraph Studio UI (if installed)
```

The `langgraph.json` configuration file defines which agents are available. You can interact with agents via the API or through LangGraph Studio's visual interface.

For more details, see the [LangGraph CLI documentation](https://docs.langchain.com/langsmith/cli#langgraph-cli).

### Model Configuration

This repository uses a **centralized utils module** (`utils/`) to avoid code duplication. All model configurations and shared utilities are defined here:
- **`utils/models.py`** - LLM model initialization (OpenAI, Anthropic, Azure, Bedrock, Vertex AI)
- **`utils/utils.py`** - Shared utility functions (`show_graph`, `get_engine_for_chinook_db`)

**Default**: OpenAI with `o3-mini` model. To switch providers, edit `utils/models.py` following the instructions below.

**Note**: Notebooks automatically add the project root to Python's path, so they can import from `utils` regardless of which subdirectory they're in.

### Azure OpenAI Instructions

If you are using Azure OpenAI instead of OpenAI, follow these steps:

1. **Set environment variables** in your `.env` file:
   ```
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_ENDPOINT=your_endpoint
   AZURE_OPENAI_API_VERSION=2024-03-01-preview
   ```

2. **Update `utils/models.py`**:
   - Comment out the "Default Models" section (lines 20-28)
   - Uncomment the "AZURE OpenAI Version" section (lines 31-57)
   - Configure the `azure_deployment` name to match your deployment

3. **Done!** All agents and notebooks will automatically use the Azure model

### AWS Bedrock Instructions

If you are using AWS Bedrock instead of OpenAI, follow these steps:

1. **Set environment variables** in your `.env` file:
   ```
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION_NAME=us-east-1
   AWS_MODEL_ARN=your_model_arn
   ```

2. **Update `utils/models.py`**:
   - Comment out the "Default Models" section (lines 20-28)
   - Uncomment the "Bedrock Version" section (lines 60-78)
   - Configure the model settings as needed

3. **Done!** All agents and notebooks will automatically use the Bedrock model

### Google Vertex AI Instructions

If you are using Google Vertex AI instead of OpenAI, follow these steps:

1. **Set up Google Cloud credentials**
   - Create a service account in your Google Cloud project with Vertex AI permissions
   - Download the service account JSON key file
   - Save it as `vertexCred.json` in the project root directory

2. **Configure environment variables** in your `.env` file:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=./vertexCred.json
   ```

3. **Update `utils/models.py`**:
   - Comment out the "Default Models" section (lines 20-28)
   - Uncomment the "Google Vertex AI version" section (lines 81-100)
   - The setup automatically handles credential paths using `Path(__file__)`

4. **Done!** All agents and notebooks will automatically use the Vertex AI model

**Note:** Make sure `vertexCred.json` is added to your `.gitignore` to avoid committing credentials.

## Getting Started

### Recommended Learning Path

1. **Start with LG101** - `notebooks/LG101/`
   - Begin with `langgraph_101.ipynb` to learn the fundamentals
   - Continue with `langgraph_102.ipynb` for middleware and human-in-the-loop patterns

2. **Progress to LG201** - `notebooks/LG201/`
   - Explore `email_agent.ipynb` for a complete stateful agent example
   - Build multi-agent systems with `multi_agent.ipynb`

3. **Run Agents Locally**
   - Check out the `agents/` directory for standalone agent implementations
   - Use `langgraph dev` to run agents as a service

### Resources

- **[LangChain Documentation](https://docs.langchain.com/oss/python/langchain/overview)** - Complete LangChain reference
- **[LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/overview)** - LangGraph guides and API reference  
- **[LangChain Academy](https://academy.langchain.com/)** - Free courses with video tutorials
- **[LangSmith](https://smith.langchain.com)** - Debugging and monitoring for LLM applications