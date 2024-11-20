import pandas as pd
from IPython.display import display
import requests
from random import randint

url = 'https://maps.googleapis.com/maps/api/streetview'
coords = pd.read_csv('random_coordinates.csv')

display(coords)

for i in len(coords.index):
    params = {
            'key': '***REMOVED***',
            'size': '640x640',
            'location': str(coords['Latitude'][i]) + ',' + str(coords['Longitude'][i]),
            'heading': str((randint(0, 3) * 90) + randint(-15, 15)),
            'pitch': '20',
            'fov': '90'
            }
        
    response = requests.get(url, params)
    
    # Save the image to the output folder
    with open(f'images/street_view_{i}.jpg', "wb") as file:
        file.write(response.content)
