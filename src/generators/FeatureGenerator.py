import os, csv
import skvideo.io
from tensorflow.keras.utils import Sequence
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16

def init_VGG16():
    vgg = VGG16(include_top=True, weights='imagenet') 
    # Remove classification layer
    return Model(inputs=vgg.input, outputs=vgg.layers[-2].output)


class FeatureGenerator(Sequence):
    def __self__(self, video_dir, csv_dir, ObjectDetectorInitializer=init_VGG16):
        self.VIDEO_DIR = video_dir

    def from_csv(self, csv_path):
        data = defaultdict(list)
        with open(csv_path, 'r') as fcsv:
            reader = csv.reader(fcsv)
            for row in reader:
                frame, ID, cls, x1, y1, x2, y2, has_collision = row

                data['frame'].append(frame)
                data['ID'].append(ID)
                data['class'].append(cls)
                data['bb'].append((x1,y1,x2,y2))
                data['collision'] = has_collision


        return data#, self.from_video(data, os.path.join(self.VIDEO_DIR, "{}.mp4".format(ID)))


    def from_video(self, data, video_path):
        if not os.path.exists(video_path): raise RuntimeError("File does not exist: {}".format(video_path))

        # TODO: load video with 20 fps
        max_num_frames = 100
        num_detections = 19
        feature_dim    = 4096

        vnpy = skvideo.io.vread(video_path)

        feature_vector = np.zeros((max_num_frames, num_detections + 1, feature_dim))

        # Crop Bounding Boxes and run vgg
        bb_index = 0
        frame = data['frame'][0]
        for i in range(len(data['frame'])):
            if (bb_index >= num_detections):
                print("Using fewer object detections ({}) than what's available.".format(num_detections))
                break

            assert data['frame'][i] >= frame
            if (data['frame'][i] > frame): bb_index = 0
            frame = data['frame'][i]
            x1, y1, x2, y2 = data['bb'][i]

            cropped_object = vnpy[frame, y1:y2, x1:x2, :]

            feature_vector[frame, bb_index, :] = vgg(cropped_object).squeeze()

            bb_index += 1


        # Run VGG on full images
        feature_vector[frame, -1, :] = vgg(vpny[frame, ...]).squeeze()

        return feature_vector
        





vgg = init_VGG16()
vgg.summary()
