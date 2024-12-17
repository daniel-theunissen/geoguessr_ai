from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

# Example data: latitudes and longitudes
data = pd.read_csv("random_coordinates.csv")
latitudes = data["Latitude"]
longitudes = data["Longitude"]


# Convert to Cartesian
def spherical_to_cartesian(lat, lon):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.vstack((x, y, z)).T


cartesian_coords = spherical_to_cartesian(latitudes, longitudes)

# Perform K-Means clustering
n_clusters = 64  # Choose number of regions
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
data["region"] = kmeans.fit_predict(cartesian_coords)

# Save clustered data
data.to_csv("clustered_coordinates.csv", index=False)
