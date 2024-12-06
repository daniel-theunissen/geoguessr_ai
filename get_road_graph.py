import osmnx as ox

ox.config(use_cache=True, log_console=True)
# Define the area (the United States)
place_name = "United States"

# Download the road network for driving
print("Downloading the US road network...")
G = ox.graph_from_place(place_name, network_type="drive")

# Save the graph to a file for future use
ox.save_graphml(G, filepath="us_road_network.graphml")

print("US road network downloaded and saved to us_road_network.graphml")