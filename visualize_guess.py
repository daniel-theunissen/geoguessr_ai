import torch
from PIL import Image
import folium
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from model import GeoLocViT

# Define preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to model input size
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])

def denormalize_coordinates(lat_norm, lon_norm, lat_range=(-90, 90), lon_range=(-180, 180)):
    """
    Denormalize latitude and longitude from the range [-1, 1] back to real-world coordinates.

    Args:
        lat_norm (float): Normalized latitude.
        lon_norm (float): Normalized longitude.
        lat_range (tuple): Latitude range (lat_min, lat_max).
        lon_range (tuple): Longitude range (lon_min, lon_max).

    Returns:
        tuple: Real-world latitude and longitude.
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    lat = (lat_norm + 1) / 2 * (lat_max - lat_min) + lat_min
    lon = (lon_norm + 1) / 2 * (lon_max - lon_min) + lon_min

    return lat, lon


def preprocess_image(image):
    """
    Preprocess an image for the model.
    Args:
        image (PIL.Image): Input image.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose([
                # Resize the image
                transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
                # Center crop to 224x224
                transforms.CenterCrop(224),
                # Convert to tensor and scale to [0.0, 1.0]
                transforms.ToTensor(),
                # Normalize using mean and std
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
    data = transform(image)
    return data.unsqueeze(0)  # Add batch dimension

def predict_location(model, image, device):
    """
    Predict the geolocation of an image using the model.
    Args:
        model (torch.nn.Module): The trained geolocation model.
        image (PIL.Image): The input image.
        device (torch.device): Device to run the model on.
    Returns:
        tuple: Predicted (latitude, longitude).
    """
    # Preprocess image
    image_tensor = preprocess_image(image).to(device)
    model.eval()  # Ensure model is in evaluation mode
    with torch.no_grad():
        output = model(image_tensor)  # Get prediction
        predicted_coords = output.squeeze().cpu().numpy()  # Convert to NumPy
        print(predicted_coords)
        #predicted_coords = denormalize_coordinates(predicted_coords[0],predicted_coords[1])
    return tuple(predicted_coords)

def visualize_prediction(image, actual_coords, predicted_coords, map_center=None):
    """
    Visualize the actual and predicted locations on an interactive map using Folium.
    Args:
        image (PIL.Image): Input image.
        actual_coords (tuple): Actual coordinates as (latitude, longitude).
        predicted_coords (tuple): Predicted coordinates as (latitude, longitude).
        map_center (tuple or None): Optional map center as (latitude, longitude). Defaults to actual_coords.
    Returns:
        folium.Map: A Folium map with visualization.
    """
    from PIL import Image
    import numpy as np

    # Convert image to PNG format for Folium popup
    img_path = "/tmp/temp_image.png"  # Temporary path to save the image
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())  # Normalize
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    image.save(img_path)

    # Map center defaults to actual location
    map_center = map_center or actual_coords

    # Create Folium map
    m = folium.Map(location=map_center, zoom_start=10)

    # Add actual location marker
    folium.Marker(
        location=actual_coords,
        popup="Actual Location",
        icon=folium.Icon(color='green', icon='check')
    ).add_to(m)

    # Add predicted location marker
    folium.Marker(
        location=predicted_coords,
        popup="Predicted Location",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)

    # Add a line connecting the two points
    folium.PolyLine(
        locations=[actual_coords, predicted_coords],
        color='blue',
        weight=2.5,
        opacity=0.7
    ).add_to(m)

    # Add image popup at actual location
    encoded = folium.Html(f'<img src="{img_path}" width="200">', script=True)
    popup = folium.Popup(encoded, max_width=2650)
    folium.Marker(location=actual_coords, popup=popup).add_to(m)

    return m

model = GeoLocViT()  # Load your trained model
model.load_state_dict(torch.load('best_vit_geolocation.pth',weights_only=True))  # Load trained weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

image_path = "output2/train/street_view_1.jpg"
input_image = Image.open(image_path).convert("RGB")

actual_coords = (41.81648264948324,-71.46706588560826) 
predicted_coords = predict_location(model, input_image, device)  # Get predicted coordinates
print(f"Predicted Coordinates: {predicted_coords}")

map_view = visualize_prediction(input_image, actual_coords, predicted_coords)
map_view.save("geolocation_prediction.html")  # Save to an HTML file





