import cPickle
import os
from os import path

import cv2

from trash import face_extractor

extractor = face_extractor.LandmarkExtractor(10)

features = {}

#dumpfile = open("D:/Master Thesis/Dataset/landmarks.pkl", 'wb')

for root, dirs, files in os.walk("D:/Master Thesis/Dataset"):
    for name in files:
        if os.path.splitext(name)[1] != '.wmv':
            continue

        print(name)

        idx_name = name.split('_')[0]
        if idx_name not in features.keys():
            features[idx_name] = {}

        video_path = path.normpath(path.join(root, name)).replace('\\', '/')

        features[idx_name][name] = {}

        cap = cv2.VideoCapture(video_path)

        frame = -1
        while True:
            ret, img = cap.read()
            frame += 1
            if not ret:
                features[idx_name][name]["num_frames"] = frame
                break

            landmarks = extractor.track_landmarks(img)

            if landmarks is not None:
                features[idx_name][name][frame] = landmarks

cPickle.dump(features, dumpfile, -1)
