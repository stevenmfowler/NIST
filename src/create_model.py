import os
import requests
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt

def get_or_download_nist_data(url: str, file_path: str):
    """
    Ensures the NIST data file is present locally, downloading it if necessary.

    Args:
        url (str): The URL of the file to download.
        file_path (str): The absolute path where the file should be saved.

    Returns:
        None: The function saves the file directly.
    """

    if os.path.exists(file_path):
        print(f"File already exists at {file_path}. Skipping download.")
        return

    # Ensure the destination directory exists before attempting to download.
    parent_dir = os.path.dirname(file_path)
    os.makedirs(parent_dir, exist_ok=True)

    print(f"Downloading data from {url}...")
    try:
        # Use a timeout for the request and check for errors
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        print(f"Saving file to {file_path}...")
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print("Download complete.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        exit(1) # Exit if the download fails, as the rest of the script cannot proceed.

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from the specified Excel file and preprocesses it.

    This function reads the first sheet of the Excel file and standardizes
    the 'Control Identifier' column by replacing parentheses with hyphens
    (e.g., 'AC-2(1)' becomes 'AC-2-1').

    Args:
        file_path (str): The absolute path to the Excel data file.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    print("Loading data from the first sheet of the Excel file...")
    # By default, read_excel loads the first sheet if sheet_name is not specified.
    controls_df = pd.read_excel(file_path)

    # Standardize the 'Control Identifier' column format
    identifier_col = 'Control Identifier'
    if identifier_col in controls_df.columns:
        print(f"Standardizing '{identifier_col}' column...")
        # Ensure column is string type to handle potential mixed types or NaNs.
        # Replace '(' with '-' and remove ')' to format e.g., AC-2(1) -> AC-2-1
        controls_df[identifier_col] = controls_df[identifier_col].astype(str).str.replace(
            '(', '-', regex=False
        ).str.replace(
            ')', '', regex=False
        )
    else:
        print(f"Warning: Column '{identifier_col}' not found. Skipping standardization.")

    return controls_df

def save_processed_data(df: pd.DataFrame, file_path: str):
    """
    Saves the processed DataFrame to a specified file format.
    The format is inferred from the file extension (e.g., .csv, .parquet).

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): The path where the file will be saved.
    """
    # Ensure the parent directory exists
    parent_dir = os.path.dirname(file_path)
    os.makedirs(parent_dir, exist_ok=True)

    print(f"Saving processed data to {file_path}...")
    try:
        df.to_csv(file_path, index=False)
        print("Data saved successfully.")
    except Exception as e:
        print(f"Error: Could not write to file at {file_path}. Check permissions and path.")
        print(f"Details: {e}")
        exit(1)

def load_processed_data(file_path: str) -> pd.DataFrame | None:
    """
    Loads a processed DataFrame from a local file if it exists.
    Supports CSV and Parquet formats, inferred from file extension.

    Args:
        file_path (str): The path to the processed data file.

    Returns:
        pd.DataFrame or None: The loaded DataFrame, or None if the file
                              doesn't exist or cannot be read.
    """
    if not os.path.exists(file_path):
        return None

    print(f"Loading processed data from {file_path}...")
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading processed file {file_path}: {e}")
        return None

def create_knowledge_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Creates a knowledge graph from the NIST controls DataFrame.

    Nodes represent controls, and edges represent hierarchical relationships
    (e.g., an edge from 'AC-2' to 'AC-2-1').

    Args:
        df (pd.DataFrame): The preprocessed controls DataFrame.

    Returns:
        nx.DiGraph: A directed graph representing the control relationships.
    """
    G = nx.DiGraph()
    identifier_col = 'Control Identifier'

    # Ensure the identifier column exists
    if identifier_col not in df.columns:
        print(f"Error: Column '{identifier_col}' not found in DataFrame. Cannot build graph.")
        return G

    # Add all controls as nodes with their data as attributes
    for index, row in df.iterrows():
        control_id = row[identifier_col]
        # Add node and unpack the row's data as node attributes
        G.add_node(control_id, **row.to_dict())

    # Add edges for hierarchical relationships by parsing control IDs
    for control_id in G.nodes():
        # Find parent by splitting the ID. e.g., 'AC-2-1' -> parent 'AC-2'
        parts = str(control_id).split('-')
        if len(parts) > 2:  # Indicates a potential enhancement, e.g., 'AC-2-1'
            parent_id = '-'.join(parts[:-1])
            if G.has_node(parent_id):
                G.add_edge(parent_id, control_id, relationship='enhancement')

    return G

def visualize_knowledge_graph(G: nx.DiGraph, nodes_to_include: list, output_path: str, title: str):
    """
    Creates and saves a visualization of a subgraph.

    Args:
        G (nx.DiGraph): The full knowledge graph.
        nodes_to_include (list): A list of node IDs to include in the visualization.
        output_path (str): The path to save the output image file.
        title (str): The title for the plot.
    """
    # Create a subgraph containing only the nodes we want to visualize
    subgraph = G.subgraph(nodes_to_include)

    if subgraph.number_of_nodes() == 0:
        print(f"Warning: Subgraph for '{title}' is empty. Skipping visualization.")
        return

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(20, 20))

    # Use a layout that's good for hierarchies if possible, otherwise spring_layout
    try:
        # `graphviz_layout` is excellent for trees but requires pygraphviz
        pos = nx.nx_agraph.graphviz_layout(subgraph, prog='dot')
    except ImportError:
        print("`pygraphviz` not found, using `spring_layout`. For better hierarchy plots, run: pip install pygraphviz")
        pos = nx.spring_layout(subgraph, seed=42, k=0.8)

    nx.draw(subgraph, pos, with_labels=True, node_size=3000, node_color='#a0cbe2',
            font_size=10, font_weight='bold', arrows=True, arrowstyle='->',
            arrowsize=20, edge_color='gray')

    plt.title(title, size=20)
    
    print(f"Saving visualization to {output_path}...")
    plt.savefig(output_path, format="PNG", bbox_inches="tight")
    plt.close() # Close the figure to free up memory

def main():
    """Main function to orchestrate the data pipeline."""
    # --- Configuration ---
    URL = "https://csrc.nist.gov/files/pubs/sp/800/53/r5/upd1/final/docs/sp800-53r5-controls.xlsx"
    RAW_FILENAME = "sp800-53r5-controls.xlsx"
    PROCESSED_FILENAME = "processed_nist_controls.csv"

    # Determine paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    raw_filepath = os.path.join(data_dir, RAW_FILENAME)
    processed_filepath = os.path.join(data_dir, PROCESSED_FILENAME)

    # --- Data Pipeline ---
    # Step 1: Try to load the already-processed data to save time
    controls_df = load_processed_data(processed_filepath)

    # Step 2: If processed data doesn't exist, create it from the raw source
    if controls_df is None:
        print("Processed data not found. Starting raw data pipeline...")
        get_or_download_nist_data(URL, raw_filepath)
        controls_df = load_and_preprocess_data(raw_filepath)
        save_processed_data(controls_df, processed_filepath)

    # --- Analysis / Model Training (Placeholder) ---
    print("\nData pipeline complete. DataFrame is ready.")
    print(controls_df.head())

    # --- Knowledge Graph Creation ---
    print("\nCreating knowledge graph from controls...")
    nist_kg = create_knowledge_graph(controls_df)
    print(f"Knowledge graph created successfully.")
    print(f" - Nodes (Controls): {nist_kg.number_of_nodes()}")
    print(f" - Edges (Relationships): {nist_kg.number_of_edges()}")

    # --- Knowledge Graph Visualization ---
    FAMILY_TO_VISUALIZE = 'AC' # Example: Access Control family
    print(f"\nVisualizing knowledge graph for the '{FAMILY_TO_VISUALIZE}' family...")

    # Select all nodes belonging to the specified family
    family_nodes = [n for n in nist_kg.nodes() if str(n).startswith(f"{FAMILY_TO_VISUALIZE}-")]

    if family_nodes:
        output_dir = os.path.join(script_dir, '..', 'output')
        output_filename = f"{FAMILY_TO_VISUALIZE}_family_graph.png"
        output_filepath = os.path.join(output_dir, output_filename)

        visualize_knowledge_graph(
            nist_kg,
            nodes_to_include=family_nodes,
            output_path=output_filepath,
            title=f"NIST Control Family: {FAMILY_TO_VISUALIZE}"
        )
    else:
        print(f"No controls found for family '{FAMILY_TO_VISUALIZE}'. Skipping visualization.")

if __name__ == "__main__":
    main()