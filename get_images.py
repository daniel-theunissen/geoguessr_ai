import pandas as pd
from IPython.display import display
import requests
from random import randint

# Define the Street View image and metadata base URLs
image_url = "https://maps.googleapis.com/maps/api/streetview"
metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"

# Load coordinates from CSV
coords = pd.read_csv("random_coordinates.csv")
display(coords)

# Loop through each coordinate
for i in range(len(coords.index)):
    # Prepare the parameters for the metadata request
    location = f"{coords['Latitude'][i]},{coords['Longitude'][i]}"
    metadata_params = {
        "key": "",  # Add your Google Maps API key here
        "location": location,
    }

    # Check Street View metadata
    metadata_response = requests.get(metadata_url, params=metadata_params)
    metadata = metadata_response.json()

    # Only proceed if Street View is available at this location
    if metadata.get("status") == "OK":
        # Prepare parameters for the image request
        image_params = {
            "key": "",  # Add your Google Maps API key here
            "size": "640x640",
            "location": location,
            "heading": str((randint(0, 3) * 90) + randint(-15, 15)),
            "pitch": "20",
            "fov": "90",
        }

        # Fetch the Street View image
        response = requests.get(image_url, params=image_params)

        # Save the image to the output folder
        with open(f"images/street_view_{i}.jpg", "wb") as file:
            file.write(response.content)
    else:
        print(f"No Street View coverage at {location}")
