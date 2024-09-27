# Setup
Follow these instructions to get ready for your upcoming session!

If you run into issues with setting up the python environment or acquiring the necessary API keys due to any resrictions (ex. corporate policy), contact your LangChain representative and we'll find a work-around!

### Create an environment and install dependencies  
```
$ cd langgraph-101
$ python3 -m venv lg-101-env
$ source lg-101-env/bin/activate
$ pip install -r requirements.txt
```

### Running notebooks
Make sure the following command works and opens the relevant notebooks
```
$ jupyter notebook
```

### Set OpenAI API key
* If you don't have an OpenAI API key, you can sign up [here](https://openai.com/index/openai-api/).
*  Set `OPENAI_API_KEY` in the .env file.

### Sign up for LangSmith

* Sign up [here](https://docs.smith.langchain.com/) 
*  Set `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2=true` .env file.

### Tavily for web search

Tavily Search API is a search engine optimized for LLMs and RAG, aimed at efficient, quick, and persistent search results. You can sign up for an API key [here](https://tavily.com/). It's easy to sign up and offers a generous free tier. Some lessons (in Module 4) will use Tavily. Set `TAVILY_API_KEY` in .env file.

### Set up LangGraph Studio

* Currently Studio only has macOS support
* Download the latest `.dmg` file [here](https://github.com/langchain-ai/langgraph-studio?tab=readme-ov-file#download)
* Install Docker desktop for Mac [here](https://docs.docker.com/engine/install/)

### Running Studio
To use Studio, you will need to fill in the separate .env file in the /studio/ folder with the relevant API keys. Once you start up LangGraph studio, you can load in the entire /studio folder from this repository.