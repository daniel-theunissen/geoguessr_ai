import argparse
import csv
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from random import randint
import numpy as np
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        help="The CSV file to read and extract GPS coordinates and cluster targets from",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--images",
        help="The path to the images folder (defaults to: images/)",
        default="images/",
        type=str,
    )
    parser.add_argument("--output", help="The output folder", required=True, type=str)
    return parser.parse_args()


args = get_args()

targets_train = []
targets_val = []


def get_data(coord, coord_index):
    """
    Retrieve the image and cluster target for a given coordinate.

    Args:
        coord (list): A row from the CSV file, expected to contain [latitude, longitude, cluster_label].
        coord_index (int): Index of the current row for naming the output image.

    Returns:
        list: The loaded image and the cluster target.
    """
    img_path = os.path.join(args.images, f"street_view_{coord_index}.jpg")
    try:
        img = Image.open(img_path)
    except (FileNotFoundError, UnidentifiedImageError) as e:
        print(f"Error loading image for coord_index {coord_index}: {e}")
        return None  # Skip this entry if the image is missing or corrupt

    try:
        cluster_label = int(coord[2])  # Extract the cluster label (third column)
    except (IndexError, ValueError):
        print(f"Invalid cluster label at coord_index {coord_index}: {coord}")
        return None  # Skip if the cluster label is invalid

    return [img, cluster_label]


def main():
    with open(args.file, "r") as f:
        coords_reader = csv.reader(f)
        coords = list(coords_reader)  # Read all rows

    train_data_path = os.path.join(args.output, "train")
    os.makedirs(train_data_path, exist_ok=True)
    val_data_path = os.path.join(args.output, "val")
    os.makedirs(val_data_path, exist_ok=True)

    val_count = 0
    train_count = 0

    for coord_index, coord in enumerate(tqdm(coords)):
        data = get_data(coord, coord_index)
        if data is None:
            continue  # Skip if no valid data was returned

        if randint(0, 9) == 0:
            # Save to validation set
            val_image_path = os.path.join(
                args.output, f"val/street_view_{val_count}.jpg"
            )
            data[0].save(val_image_path)
            targets_val.append(data[1])
            val_count += 1
        else:
            # Save to training set
            train_image_path = os.path.join(
                args.output, f"train/street_view_{train_count}.jpg"
            )
            data[0].save(train_image_path)
            targets_train.append(data[1])
            train_count += 1

    # Save targets as numpy arrays
    np.save(os.path.join(args.output, "train/targets.npy"), np.array(targets_train))
    np.save(os.path.join(args.output, "val/targets.npy"), np.array(targets_val))

    print("Train Files:", train_count)
    print("Val Files:", val_count)


if __name__ == "__main__":
    main()
