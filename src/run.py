#!/usr/bin/env python3
import os
import sys
from argparse import ArgumentParser
from zipfile import ZipFile
from model.DSARNN import DSARNN
import numpy as np

class CLIParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
            description="A replication of 'Anticipating Accidents in Dashcam Videos'")
        self.add_argument('--features_file', default=None, type=str,
		help="Path to the features.zip file")

if __name__ == '__main__':
    parser = CLIParser()
    args = parser.parse_args()
    features_file = args.features_file


    ########### Global Parameters ###########
    train_num = 126
    test_num = 46
    learning_rate = 0.0001
    n_epochs = 30
    batch_size = 10
    display_step = 10

    ########### Network Parameters ###########
    n_input = 4096
    n_detection = 20
    n_hidden = 512
    n_img_hidden = 256
    n_att_hidden = 256
    n_classes = 2
    n_frames = 100

    if not os.path.exists('./features'):
        if features_file is None:
            print("ERROR: You must specify --features_file to unzip the features")
            sys.exit(1)
        if not os.path.exists(features_file):
            print(f"ERROR: Could not find the file '{features_file}'")
            sys.exit(1)
        
        print("Extracting features...")
        ZipFile(features_file).extractall()
        print("Extracting done!")
    
    # Sanity check...
    for i in range(1, train_num):
        file_name = f"./features/training/batch_{i:03d}.npz"
        if not os.path.exists(file_name):
            print(f"ERROR: Could not find batch {file_name}")
            sys.exit(1)
    
    model = DSARNN(n_features=256)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    for i in range(1, train_num):
        file_name = f"./features/training/batch_{i:03d}.npz"
        batch_data = np.load(file_name)
        X = batch_data['data']
        Y = batch_data['labels']
        model.fit(X, Y, epochs=n_epochs)
