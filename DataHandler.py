import scipy.io as sio
import os
import pickle

def load_data(directory_path = "images/all_images_", mat_file_path = "GPS_Long_Lat_Compass.mat"):
    images_by_id = {}
    images_by_coordinates = {}
    path_to_coordinates = {}

    # Load GPS data
    mat_contents = sio.loadmat(mat_file_path)
    if 'GPS_Compass' in mat_contents:
        gps_data = mat_contents['GPS_Compass']
    coordinate_values = list(zip(gps_data[:, 0], gps_data[:, 1]))

    # Process images and coordinates
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg"):
            image_id = filename.split("_")[0]
            if image_id in images_by_id:
                images_by_id[image_id].append(os.path.join(directory_path, filename))
            else:
                images_by_id[image_id] = [os.path.join(directory_path, filename)]

    for index, coords in enumerate(coordinate_values):
        image_id = str(index + 1).zfill(6)
        if image_id in images_by_id:
            images_by_coordinates[coords] = images_by_id[image_id]

    for image_id, images in images_by_id.items():
        coords = coordinate_values[int(image_id) - 1]
        for image_path in images:
            path_to_coordinates[image_path] = coords

    return images_by_id, images_by_coordinates, path_to_coordinates

def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_saved_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)