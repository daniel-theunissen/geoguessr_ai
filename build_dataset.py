import argparse
import csv
from PIL import Image
from tqdm import tqdm
from random import randint
import numpy as np
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="The CSV file to read and extract GPS coordinates from", required=True, type=str)
    parser.add_argument("--images", help="The path to the images folder, (defaults to: images/)", default='images/', type=str)
    parser.add_argument("--output", help="The output folder", required=True, type=str)
    return parser.parse_args()

args = get_args()

targets_train = []
targets_val = []

# def multi_label(num, decimals=4):
#     num_original = str(abs(float(num)))
    
#     if decimals > 0:
#       num = str(round(float(num) * (10 ** decimals)))
#     else:
#       num = str(round(float(num)))
    
#     if num[0] == '-':
#         label = np.array([1])
#         num = num[1:]
#     else:
#         label = np.array([0])
    
#     num = num.ljust(len(num_original.split('.')[0]) + decimals, '0')
#     num = num.zfill(decimals + 3)
    
#     for digit in num:
#         label = np.concatenate((label, np.eye(10)[int(digit)]))
    
#     return label

def get_data(coord, coord_index):
    lat, lon = float(coord[0]), float(coord[1])
    
    img_path = os.path.join(args.images, f'street_view_{coord_index}.jpg')
    img = Image.open(img_path)
    
    # lat_multi_label = multi_label(lat)
    # lon_multi_label = multi_label(lon)

    # print(f"lat_multi_label: {lat_multi_label}")
    # print(f"lon_multi_label: {lon_multi_label}")
    
    # target = np.concatenate((lat_multi_label, lon_multi_label))
    lat_norm, lon_norm = normalize_coordinates(lat,lon)
    target = np.concatenate((np.array([lat]), np.array([lon])))

    # print(target)
    # print(type(target))
    
    return [img, target]

def normalize_coordinates(lat, lon, lat_range=(-90, 90), lon_range=(-180, 180)):
    """
    Normalize latitude and longitude to the range [-1, 1].

    Args:
        lat (float): Latitude value.
        lon (float): Longitude value.
        lat_range (tuple): Latitude range (lat_min, lat_max).
        lon_range (tuple): Longitude range (lon_min, lon_max).

    Returns:
        tuple: Normalized latitude and longitude.
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    lat_norm = 2 * (lat - lat_min) / (lat_max - lat_min) - 1
    lon_norm = 2 * (lon - lon_min) / (lon_max - lon_min) - 1

    return lat_norm, lon_norm

def main():
    with open(args.file, 'r') as f:
        coords_reader = csv.reader(f)
        coords = []
        for row in coords_reader:
            coords.append(row)

    
    train_data_path = os.path.join(args.output, 'train')
    os.makedirs(train_data_path, exist_ok=True)
    val_data_path = os.path.join(args.output, 'val')
    os.makedirs(val_data_path, exist_ok=True)
    
    val_count = 0
    train_count = 0
    
    for coord_index, coord in enumerate(tqdm(coords)):
        if randint(0, 9) == 0:
            data = get_data(coord, coord_index)
            #print(f"coord: {coord}")

            val_data_path = os.path.join(args.output, f'val/street_view_{val_count}.jpg')
            data[0].save(val_data_path)
            targets_val.append(data[1])
            val_count += 1
        else:
            data = get_data(coord, coord_index)
            train_data_path = os.path.join(args.output, f'train/street_view_{train_count}.jpg')
            data[0].save(train_data_path)
            targets_train.append(data[1])
            train_count += 1
    
    np.save(os.path.join(args.output, f'train/targets.npy'), np.array(targets_train))
    np.save(os.path.join(args.output, f'val/targets.npy'), np.array(targets_val))
    
    print('Train Files:', train_count)
    print('Val Files:', val_count)

if __name__ == '__main__':
    main()