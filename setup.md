# Setup - Google Vertex AI Branch
Follow these instructions to get ready for your upcoming session with Google Vertex AI!

If you run into issues with setting up the python environment or acquiring the necessary credentials due to any restrictions (ex. corporate policy), contact your LangChain representative and we'll find a work-around!

### Create an environment and install dependencies  
```bash
# Ensure you have a recent version of pip and python installed
$ cd langgraph-101
$ git checkout vertex
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

### Running notebooks
Make sure the following command works and opens the relevant notebooks
```bash
$ jupyter notebook
```

### Set up Google Vertex AI credentials

1. **Create a Google Cloud Project** (if you don't have one):
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one

2. **Enable Vertex AI API**:
   - In the Google Cloud Console, go to APIs & Services → Library
   - Search for "Vertex AI API" and enable it

3. **Create a Service Account**:
   - Go to IAM & Admin → Service Accounts
   - Click "Create Service Account"
   - Give it a name (e.g., "langgraph-vertex")
   - Grant the following roles:
     - Vertex AI User
     - AI Platform Developer

4. **Download credentials**:
   - Click on your service account
   - Go to "Keys" tab → "Add Key" → "Create New Key" → "JSON"
   - Download the JSON file and save it securely

5. **Set up environment variables**:
   - Create a `.env` file in the project root
   - Add the path to your credentials file:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/vertex-credentials.json
   ```

### [Optional] Sign up for LangSmith

* Sign up [here](https://docs.smith.langchain.com/) 
* Set `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, and `LANGSMITH_PROJECT` in the .env file.

**Note:** This setup is specifically for the Vertex AI branch. For other providers (OpenAI, Azure OpenAI), please use the main branch.