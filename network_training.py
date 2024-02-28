import numpy as np
import csv

from network.calibration_network import training
from config import params

if __name__ == "__main__":
    # Read corners from csv file
    filename = params.save_corners_path + '/' + params.save_corners_file + '.csv'
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        corners = []
        for row in reader:
            corner = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])]
            corners.append(corner)
    corners=np.asarray(corners)

    outputs_file = params.save_path + '/' + params.outputs_file + '.csv'
    with open(outputs_file, newline='') as f:
        reader = csv.reader(f)
        outputs = []
        for row in reader:
            output = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])]
            outputs.append(output)
    outputs=np.asarray(outputs)

    training(corners, outputs)