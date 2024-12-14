import torch
from PIL import Image
import folium
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import pandas as pd
from model import GeoLocViT, GeoLocResNet
from torchvision.models import wide_resnet50_2
import random
import os
import numpy as np

# Ensure reproducibility of random selection
random.seed(546)

# Load the region-to-coordinates mapping
region_coords = pd.read_csv("clustered_coordinates_fix.csv")


def preprocess_image(image):
    """
    Preprocess an image for the model.
    Args:
        image (PIL.Image): Input image.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data = transform(image)
    return data.unsqueeze(0)  # Add batch dimension


def predict_region(model, image, device):
    """
    Predict the region index for an image using the classification model.
    Args:
        model (torch.nn.Module): The trained geolocation model.
        image (PIL.Image): The input image.
        device (torch.device): Device to run the model on.
    Returns:
        int: Predicted region index.
    """
    image_tensor = preprocess_image(image).to(device)
    model.eval()  # Ensure model is in evaluation mode
    with torch.no_grad():
        output = model(image_tensor)  # Get region logits
        predicted_region_index = (
            output.argmax().item()
        )  # Get the region with highest probability
    return predicted_region_index


def get_coordinates_from_region(region_index):
    """
    Map the predicted region index to latitude and longitude.
    Args:
        region_index (int): The predicted region index.
    Returns:
        tuple: Corresponding (latitude, longitude) of the predicted region.
    """
    region_data = region_coords[region_coords["region"] == region_index].iloc[0]
    latitude = region_data["Latitude"].mean()
    longitude = region_data["Longitude"].mean()
    return latitude, longitude


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

    img_path = "/tmp/temp_image.png"  # Temporary path to save the image
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())  # Normalize
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    image.save(img_path)

    map_center = map_center or actual_coords

    # Create Folium map
    m = folium.Map(location=map_center, zoom_start=10)

    # Add actual location marker
    folium.Marker(
        location=actual_coords,
        popup="Actual Location",
        icon=folium.Icon(color="green", icon="check"),
    ).add_to(m)

    # Add predicted location marker
    folium.Marker(
        location=predicted_coords,
        popup="Predicted Location",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    # Add a line connecting the two points
    folium.PolyLine(
        locations=[actual_coords, predicted_coords],
        color="blue",
        weight=2.5,
        opacity=0.7,
    ).add_to(m)

    # Add image popup at actual location
    encoded = folium.Html(f'<img src="{img_path}" width="200">', script=True)
    popup = folium.Popup(encoded, max_width=2650)
    folium.Marker(location=actual_coords, popup=popup).add_to(m)

    return m


def visualize_prediction_two_models(
    image, actual_coords, predicted_coords1, predicted_coords2, map_center=None
):
    """
    Visualize the actual and predicted locations from two models on an interactive map using Folium.
    Args:
        image (PIL.Image): Input image.
        actual_coords (tuple): Actual coordinates as (latitude, longitude).
        predicted_coords1 (tuple): Predicted coordinates from model1 as (latitude, longitude).
        predicted_coords2 (tuple): Predicted coordinates from model2 as (latitude, longitude).
        map_center (tuple or None): Optional map center as (latitude, longitude). Defaults to actual_coords.
    Returns:
        folium.Map: A Folium map with visualization.
    """
    img_path = "/tmp/temp_image.png"  # Temporary path to save the image
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())  # Normalize
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    image.save(img_path)

    map_center = map_center or actual_coords

    # Create Folium map
    m = folium.Map(location=map_center, zoom_start=10)

    # Add actual location marker
    folium.Marker(
        location=actual_coords,
        popup="Actual Location",
        icon=folium.Icon(color="green", icon="check"),
    ).add_to(m)

    # Add predicted location markers
    folium.Marker(
        location=predicted_coords1,
        popup="Predicted Location (Model 1)",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    folium.Marker(
        location=predicted_coords2,
        popup="Predicted Location (Model 2)",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)

    # Add lines connecting actual to predictions
    folium.PolyLine(
        locations=[actual_coords, predicted_coords1],
        color="red",
        weight=2.5,
        opacity=0.7,
    ).add_to(m)

    folium.PolyLine(
        locations=[actual_coords, predicted_coords2],
        color="blue",
        weight=2.5,
        opacity=0.7,
    ).add_to(m)

    # Add image popup at actual location
    encoded = folium.Html(f'<img src="{img_path}" width="200">', script=True)
    popup = folium.Popup(encoded, max_width=2650)
    folium.Marker(location=actual_coords, popup=popup).add_to(m)

    return m


def predict_and_visualize_on_single_map(model1, model2, device, random_images):
    """
    Predict geolocations for multiple images and visualize all predictions on a single map.
    Args:
        model (torch.nn.Module): The trained geolocation model.
        device (torch.device): Device to run the model on.
        random_images (list): List of randomly selected image numbers.
    """
    map_center = [0, 0]  # Initial map center, adjusted later based on first prediction
    m = None
    results = []
    img_pd = pd.read_csv("identical_images.csv")

    for img_num in random_images:

        val_img = img_pd.iat[img_num, 0]
        print(val_img)
        val_img_num = int(val_img.split("_")[-1].split(".")[0])
        print(val_img_num)
        image_path = f"images/{val_img}"

        if not os.path.exists(image_path):
            print(f"Image {img_num} not found at {image_path}. Skipping...")
            continue

        img = Image.open(image_path).convert("RGB")
        img_path = f"/tmp/temp_image_{img_num}.png"  # Temporary path to save the image
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min())  # Normalize
            img = (img * 255).astype(np.uint8)
            img = img.fromarray(img)
        img.save(img_path)

        input_image = Image.open(image_path).convert("RGB")

        # Actual coordinates for comparison
        actual_coords = (
            region_coords.iat[val_img_num, 0],
            region_coords.iat[val_img_num, 1],
        )

        # Predict region index using the classification model
        predicted_region_index1 = predict_region(model1, input_image, device)
        predicted_region_index2 = predict_region(model2, input_image, device)

        # Get the predicted coordinates by mapping the region index
        predicted_coords1 = get_coordinates_from_region(predicted_region_index1)
        predicted_coords2 = get_coordinates_from_region(predicted_region_index2)

        # Initialize the map if it's the first iteration
        if m is None:
            map_center = actual_coords
            m = folium.Map(location=map_center, zoom_start=2)

        # Add actual location marker

        # Add image popup at actual location
        encoded = folium.Html(f'<img src="{img_path}" width="200">', script=True)
        popup = folium.Popup(encoded, max_width=2650)
        folium.Marker(location=actual_coords, popup=popup).add_to(m)

        # Add predicted location markers
        folium.Marker(
            location=predicted_coords1,
            popup="Predicted Location (Model 1)",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

        folium.Marker(
            location=predicted_coords2,
            popup="Predicted Location (Model 2)",
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(m)

        # Add lines connecting actual to predictions
        folium.PolyLine(
            locations=[actual_coords, predicted_coords1],
            color="red",
            weight=2.5,
            opacity=0.7,
        ).add_to(m)

        folium.PolyLine(
            locations=[actual_coords, predicted_coords2],
            color="blue",
            weight=2.5,
            opacity=0.7,
        ).add_to(m)

        results.append(
            {
                "image_number": img_num,
                "actual_coords": actual_coords,
                "predicted_coords 1": predicted_coords1,
                "predicted_coords 2": predicted_coords2,
            }
        )

    # Save the map to an HTML file
    os.makedirs("predictions", exist_ok=True)
    map_output_path = "predictions/all_predictions_map.html"
    if m is not None:
        m.save(map_output_path)
        print(f"All predictions map saved at {map_output_path}")

    return results


# Load model and weights
model1 = GeoLocViT(64)  # Initialize your model
model1.load_state_dict(
    torch.load("best_vit_geolocation.pth", weights_only=True)
)  # Load trained model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1.to(device)

# Load model and weights
model2 = GeoLocResNet(64)  # Initialize your model
model2.load_state_dict(
    torch.load("best_wideres_geolocation.pth", weights_only=True)
)  # Load trained model weights
model2.to(device)

# # 14, 47, 51, 65, 79, 10026, 9238
# img_num = 9238
# image_path = f"images/street_view_{img_num}.jpg"
# input_image = Image.open(image_path).convert("RGB")

# # Actual coordinates for comparison
# actual_coords = (
#     region_coords.loc[img_num - 1, "Latitude"],
#     region_coords.loc[img_num - 1, "Longitude"],
# )

# # Predict region index using the classification model
# predicted_region_index1 = predict_region(model1, input_image, device)
# predicted_region_index2 = predict_region(model2, input_image, device)

# # Get the predicted coordinates by mapping the region index
# predicted_coords1 = get_coordinates_from_region(predicted_region_index1)
# predicted_coords2 = get_coordinates_from_region(predicted_region_index2)

# # Print predicted coordinates
# print(f"Predicted Coordinates (Model 1): {predicted_coords1}")
# print(f"Predicted Coordinates (Model 2): {predicted_coords2}")

# # Visualize predictions on a map
# map_view = visualize_prediction_two_models(
#     input_image, actual_coords, predicted_coords1, predicted_coords2
# )
# map_view.save("geolocation_prediction_two_models.html")  # Save to an HTML file


# Select 15 random images from the dataset
num_images = 1500  # Total number of images in the dataset
random_images = random.sample(range(1, num_images + 1), 5)


# Run predictions and visualize on a single map
results = predict_and_visualize_on_single_map(model1, model2, device, random_images)

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("predictions/results.csv", index=False)
print("Predictions completed and results saved to predictions/results.csv")
