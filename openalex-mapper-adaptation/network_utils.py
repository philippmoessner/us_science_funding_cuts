import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from edgebundling import run_and_plot_bundling
from matplotlib.colors import Normalize

def create_citation_graph(df):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes (papers) to the graph with their positions
    pos = {}  # Dictionary to store positions
    for idx, row in df.iterrows():
        G.add_node(
            row['id'],
            X=row['x'],
            Y=row['y'],
            publication_year=row['publication_year'],
            color=row['color']
        )
        pos[row['id']] = (row['x'], row['y'])

    # Add edges based on references
    for idx, row in df.iterrows():
        source_id = row['id']
        refs = row['referenced_works']
        if isinstance(refs, list):
            references = refs
        elif isinstance(refs, str):
            references = refs.split(', ')
        else:
            references = []
        for ref in references:
            if ref in df['id'].values:
                G.add_edge(source_id, ref)
    
    G = G.to_undirected()
    return G

def draw_citation_graph(G, bundle_edges=False, path=None, min_max_coordinates=None, node_colors=None):
    pos = {}
    for node in G.nodes():
        pos[node] = (G.nodes[node]['X'], G.nodes[node]['Y'])
    fig, ax = plt.subplots(figsize=(20, 20))
    # Ensure a solid white background (no transparency)
    fig.patch.set_facecolor('green')
    ax.set_facecolor('green')
    plt.margins(0, 0)  # Remove margins
    if bundle_edges:
        # Turning color into rgb
        node_colors = {node: tuple(int(G.nodes[node]['color'].lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)) + (1.0,) for node in G.nodes()}
        
        for u, v in G.edges():
            x1, y1 = G.nodes[u]['X'], G.nodes[u]['Y']
            x2, y2 = G.nodes[v]['X'], G.nodes[v]['Y']
            G[u][v]['dist'] = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
        
        run_and_plot_bundling(G, method="hammer", ax=ax, edge_gradient=True, 
                            node_colors=node_colors, linewidths=.8, alpha=.5)
    else:
        nx.draw(G, pos=pos, node_size=0, with_labels=False, edge_color='#f98e31', alpha=0.3)
    
    plt.axis('off')
    plt.gca().set_aspect('equal')
    if min_max_coordinates is not None:
        plt.xlim(min_max_coordinates[0], min_max_coordinates[1])
        plt.ylim(min_max_coordinates[2], min_max_coordinates[3])
    
    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=800, transparent=False)
