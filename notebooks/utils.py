import os
from IPython.display import Image, display
from pathlib import Path


def get_vertex_model(model_name="gemini-2.5-flash", **kwargs):
    """Get a configured ChatVertexAI model with credentials.
    
    Automatically finds and loads credentials from project root.
    
    Args:
        model_name: The Vertex AI model to use (default: gemini-2.5-flash)
        **kwargs: Additional arguments to pass to ChatVertexAI
        
    Returns:
        Configured ChatVertexAI model
    """
    from dotenv import load_dotenv
    from langchain_google_vertexai import ChatVertexAI
    
    # Find project root
    project_root = Path(__file__).parent.parent
    
    # Load .env
    load_dotenv(dotenv_path=project_root / ".env", override=True)
    
    # Fix GOOGLE_APPLICATION_CREDENTIALS to absolute path
    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        cred_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        if not os.path.isabs(cred_path):
            absolute_path = str(project_root / cred_path.lstrip("./"))
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = absolute_path
    
    # Return configured model
    return ChatVertexAI(model=model_name, **kwargs)


def show_graph(graph, xray=False):
    """Display a LangGraph mermaid diagram with ASCII fallback.
    
    Args:
        graph: The LangGraph object that has a get_graph() method
        xray: Whether to show the internal structure of the graph
    """
    from IPython.display import Image
    try:
        return Image(graph.get_graph(xray=xray).draw_mermaid_png())
    except Exception as e:
        print(f"⚠️  Image rendering failed: {e}")
        print("\n📊 Showing ASCII diagram instead:\n")
        ascii_diagram = graph.get_graph(xray=xray).draw_ascii()
        print(ascii_diagram)
        return None