import os
import osmnx as ox
import pandas as pd

# Configure osmnx
# ox.config(use_cache=True, log_console=True)

# Top-level directory containing subfolders with .graphml files
input_directory = "graphs"  # Replace with the actual directory path
output_file = "random_coordinates.csv"

# Create a list to hold all the coordinates
all_coordinates = []


# Function to recursively find all .graphml files
def find_graphml_files(directory):
    graphml_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".graphml"):
                graphml_files.append(os.path.join(root, file))
    return graphml_files


# Get a list of all .graphml files in the directory and subdirectories
graphml_files = find_graphml_files(input_directory)

# Process each .graphml file
for filepath in graphml_files:

    print(f"Processing: {filepath}")

    G = ox.load_graphml(filepath=filepath)
    # Print CRS for debugging
    crs = G.graph.get("crs", None)

    print(f"Original CRS for {filepath}: {crs}")
    # Handle legacy CRS or invalid CRS
    if crs == {"init": "epsg:4326"} or crs == "{'init': 'epsg:4326'}":
        print(f"Converting legacy CRS for {filepath} to modern format.")
        G.graph["crs"] = "EPSG:4326"  # Convert to modern format

    try:
        # Project the graph to lat/lon
        Gp = ox.project_graph(G, to_latlong=True)

        # Convert to undirected graph
        Gp_undirected = ox.convert.to_undirected(Gp)

        # Generate random points
        points = ox.utils_geo.sample_points(
            Gp_undirected, 1
        )  # Adjust the number if needed

        # Extract coordinates
        coordinates = [(point.y, point.x) for point in points]  # (latitude, longitude)
        all_coordinates.extend(coordinates)

    except Exception as e:
        print(f"Error processing {filepath}: {e}")

# Create a DataFrame from all the coordinates
df = pd.DataFrame(all_coordinates, columns=["Latitude", "Longitude"])

# Export the DataFrame to a CSV file
df.to_csv(output_file, index=False)

print(f"Coordinates from all .graphml files have been exported to {output_file}")
