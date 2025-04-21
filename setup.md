# Setup
Follow these instructions to get ready for your upcoming session!

If you run into issues with setting up the python environment or acquiring the necessary API keys due to any resrictions (ex. corporate policy), contact your LangChain representative and we'll find a work-around!

### Create an environment and install dependencies  
```
$ cd langgraph-101
$ python3 -m venv venv
$ source venv/bin/activate
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

### [Optional] Sign up for LangSmith

* Sign up [here](https://docs.smith.langchain.com/) 
*  Set `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, and `LANGSMITH_PROJECT` in the .env file.