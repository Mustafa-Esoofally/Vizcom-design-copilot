from process_images import DesignKnowledgeGraph

def main():
    print("Starting graph initialization...")
    # Initialize the knowledge graph
    graph = DesignKnowledgeGraph()
    print("Graph initialized.")
    
    print("Generating visualization...")
    # Visualize the graph
    graph.visualize("design_knowledge_graph.png")
    print("Visualization saved to design_knowledge_graph.png")
    
    # Print graph statistics
    print("\nGraph Statistics:")
    print(f"Number of nodes: {len(graph.graph.nodes)}")
    print(f"Number of edges: {len(graph.graph.edges)}")
    
    # Print node categories
    print("\nNodes by category:")
    categories = {}
    for node, attrs in graph.graph.nodes(data=True):
        print(f"Processing node: {node} with attributes: {attrs}")
        cat = attrs.get('category', 'unknown')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(node)
    
    for cat, nodes in categories.items():
        print(f"\n{cat.upper()}:")
        for node in nodes:
            print(f"  - {node}")
    
    # Print relationships for key elements
    test_elements = ["Material", "Form", "Fire", "Water", "Performance"]
    for element in test_elements:
        print(f"\nRelationships for {element}:")
        try:
            related = graph.get_related_elements(element)
            for rel_elem, strength in related:
                # Get edge attributes
                edge_attrs = graph.graph.get_edge_data(element, rel_elem) or {}
                rel_type = edge_attrs.get('type', 'unknown')
                print(f"  - {rel_elem} (strength: {strength:.2f}, type: {rel_type})")
        except Exception as e:
            print(f"Error processing relationships for {element}: {str(e)}")

if __name__ == "__main__":
    try:
        print("Starting main function...")
        main()
        print("Main function completed.")
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise 