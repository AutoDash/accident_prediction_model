from src.generators.FeatureGenerator import FeatureGenerator
import numpy as np

generated_feature_dir = './features'
ground_truth_npz_dir = './gt_npz'

DEBUG=False

max_num_frames = 100
num_detections = 19
feature_dim = 4096



class MockObjectDetector():
    def predict(self, data):
        return np.random.random((data.shape[0], feature_dim))

fg = FeatureGenerator(max_num_frames=max_num_frames,
                      num_detections=num_detections,
                      feature_dim=feature_dim,
                      ObjectDetectorInitializer=MockObjectDetector,
                      DEBUG=DEBUG)

fg.generate_smallcorgi_features(video_dir='./videos',
                                csv_dir='./csvs',
                                out_dir=generated_feature_dir,
                                csv_delimiter='\t')






