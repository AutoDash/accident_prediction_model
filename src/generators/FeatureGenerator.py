import os, csv
import random
import numpy as np
import skimage.transform
import skvideo.io
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg_input
from tqdm import tqdm

def init_VGG16():
    vgg = VGG16(include_top=True, weights='imagenet') 
    # Remove classification layer
    return Model(inputs=vgg.input, outputs=vgg.layers[-2].output)


class FeatureGenerator():
    def __init__(self, max_num_frames, num_detections, feature_dim,  ObjectDetectorInitializer=init_VGG16, DEBUG=False):
        self.max_num_frames = max_num_frames
        self.num_detections = num_detections
        self.feature_dim   = feature_dim
        self.vgg = ObjectDetectorInitializer()
        self.DEBUG = DEBUG

    # Extract annotations from csv
    def from_csv(self, csv_path, csv_delimiter=','):
        data = defaultdict(list)
        with open(csv_path, 'r') as fcsv:
            reader = csv.reader(fcsv, delimiter=csv_delimiter)
            header = next(reader)
            for row in reader:
                frame, ID, cls, x1, y1, x2, y2, has_collision = row

                data['frame'].append(int(frame))
                data['ID'].append(ID)
                data['class'].append(cls)
                data['bb'].append((int(x1),int(y1),int(x2),int(y2)))
                data['has_collision'].append(int(has_collision))

        return data

    # Extract spatial features from a single video
    def from_video(self, data, video_path):
        if not os.path.exists(video_path): raise RuntimeError("File does not exist: {}".format(video_path))
        vnpy = skvideo.io.vread(video_path)
        vf, vh, vw, vd = vnpy.shape

        if self.DEBUG:
            print("Before padding: ", vnpy.shape)

        # Pad start of video with 0s to force 100 frames
        pad_len = self.max_num_frames - vf
        if pad_len:
            zero_padding = np.zeros((pad_len, vh, vw, vd))
            vnpy = np.concatenate((zero_padding, vnpy), axis=0)

        vnpy = vnpy.astype(np.int)

        if self.DEBUG:
            print("After padding: ", vnpy.shape)

        # Reorganize bbs to remove holes
        frame_bbs = [[] for _ in range(self.max_num_frames)]
        for frame_with_bb, bb in zip(data['frame'], data['bb']):
            frame_bbs[frame_with_bb+pad_len].append(bb)

        feature_vector = np.zeros((self.max_num_frames, self.num_detections + 1, self.feature_dim))
        det_vector     = np.zeros((self.max_num_frames, self.num_detections, 4))
        objects        = np.zeros((self.max_num_frames, self.num_detections + 1, 224, 224, 3))

        # Crop Bounding Boxes and run vgg
        for frame in range(self.max_num_frames):
            for i, (x1, y1, x2, y2) in enumerate(frame_bbs[frame]):
                if (i >= self.num_detections):
                    print("Using fewer object detections ({}) than what's available.".format(self.num_detections))
                    break

                cropped_object = vnpy[frame, y1:y2, x1:x2, :]
                cropped_object = skimage.transform.resize(cropped_object, (224,224), preserve_range=True).astype(np.int)

                if self.DEBUG: 
                    print(f"FRAME: {frame}, BBs: {x1}, {y1}, {x2}, {y2}")
                    plt.imshow(cropped_object)
                    plt.show()

                cropped_object = cropped_object[np.newaxis, ...]
                cropped_object = preprocess_vgg_input(cropped_object)

                objects[frame, i+1, ...] = cropped_object
                det_vector[frame, i, :] = [x1, y1, x2, y2]

            # Run VGG on full images
            full_frame = vnpy[frame, ...]
            full_frame = skimage.transform.resize(full_frame, (224,224), preserve_range=True).astype(int)
            full_frame = full_frame[np.newaxis, ...]
            full_frame = preprocess_vgg_input(full_frame)

            objects[frame, 0, ...] = full_frame


        feature_vector = self.vgg.predict( \
                            objects.reshape(self.max_num_frames * (self.num_detections+1), 224, 224, 3)) \
                        .reshape(feature_vector.shape)

        return feature_vector, det_vector


    def generate_smallcorgi_features(self, video_dir, csv_dir, out_dir='./npz_data', batch_size=10, shuffle=True, csv_delimiter=','):
        smallcorgi_features = np.zeros((batch_size, self.max_num_frames, self.num_detections + 1, self.feature_dim))
        smallcorgi_labels   = np.zeros((batch_size, 2))
        smallcorgi_dets     = np.zeros((batch_size, self.max_num_frames, self.num_detections, 4))
        smallcorgi_ids      = np.zeros((batch_size), dtype=str)

        if not os.path.exists(out_dir):
            os.system('mkdir -p {}'.format(out_dir))

        csv_files = os.listdir(csv_dir)
        if shuffle: random.shuffle(csv_files)

        for i, csv_fn in enumerate(tqdm(csv_files), 1):
            csv_p = os.path.join(csv_dir, csv_fn)
            vid_p = os.path.join(video_dir, "{}.mp4".format(os.path.splitext(csv_fn)[0]))

            csv_f = self.from_csv(csv_p, csv_delimiter=csv_delimiter)
            vid_f, vid_det = self.from_video(csv_f, vid_p)

            has_collision = sum(csv_f['has_collision']) != 0

            smallcorgi_features[i % batch_size, ...] = vid_f
            smallcorgi_dets[i % batch_size, ...]     = vid_det
            smallcorgi_labels[i % batch_size, ...]   = [1-has_collision, has_collision]
            smallcorgi_ids[i % batch_size, ...]      = os.path.splitext(csv_fn)[0]

            # Save batch in .npz
            if (i % batch_size == 0):
                batch_path = os.path.join(out_dir, "batch_{:05}".format(i // batch_size))
                np.savez(batch_path,
                         det=smallcorgi_dets,
                         labels=smallcorgi_labels,
                         data=smallcorgi_features,
                         ID=smallcorgi_ids)


