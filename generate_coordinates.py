import osmnx as ox
import pandas as pd

ox.config(use_cache=True, log_console=True)
G = ox.load_graphml(filepath="us_road_network.graphml")
Gp = ox.project_graph(G, to_latlong=True)
points = ox.utils_geo.sample_points(ox.convert.to_undirected(Gp), 20) # generate 20 random points

# Extract coordinates from the points
coordinates = [(point.y, point.x) for point in points]  # (latitude, longitude)

# Create a DataFrame
df = pd.DataFrame(coordinates, columns=["Latitude", "Longitude"])

# Export to CSV
output_file = "random_coordinates.csv"
df.to_csv(output_file, index=False)

print(f"Coordinates have been exported to {output_file}")