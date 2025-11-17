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