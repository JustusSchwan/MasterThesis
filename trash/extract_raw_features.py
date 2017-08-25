import os
from os import path

import cv2
import face_extractor
import numpy as np
import utility_geometric

from trash import utility_positional, filter


def draw_face(image, pts, color, thickness=1):
    open_ranges = [[0, 17], [17, 22], [22, 27], [27, 31], [31, 36]]
    closed_ranges = [[36, 42], [42, 48], [48, 60], [60, 68]]
    closed = False
    for k_range in (open_ranges, closed_ranges):
        for i_range in k_range:
            cv2.polylines(image, [np.array(pts[i_range[0]:i_range[1], :], np.int32).reshape((-1, 1, 2))], closed, color,
                          thickness)
        closed = True


extractor = face_extractor.LandmarkExtractor(10)
poseExtractor = utility_positional.poseExtractor()

features = {}

# dumpfile = open("D:/Master Thesis/Dataset/features_raw.pkl", 'wb')

kill_all = False

rotations = filter.ButterFilterArray(n=3, n_filter=2, cutoff=0.1)
for root, dirs, files in os.walk("D:/Master Thesis/Dataset"):
    if kill_all:
        break
    for name in files:
        if kill_all:
            break
        if os.path.splitext(name)[1] != '.wmv':
            continue

        print(name)

        idx_name = name.split('_')[0]
        if idx_name not in features.keys():
            features[idx_name] = {}

        video_path = path.normpath(path.join(root, 'changsun_1537.wmv')).replace('\\', '/')
        #video_path = path.normpath(path.join(root, name)).replace('\\', '/')

        features[idx_name][name] = {}

        cap = cv2.VideoCapture(video_path)

        frame = -1
        while True:
            ret, img = cap.read()
            frame += 1
            if not ret:
                break

            landmarks = extractor.track_landmarks(img)

            if landmarks is not None:
                draw_face(img, landmarks, (0, 255, 0))
                warped = extractor.warp_landmarks(landmarks) * min(img.shape[0:2])
                draw_face(img, warped, (0, 255, 255))

                pose = poseExtractor.get_head_rotation(landmarks, img.shape)

                positional = poseExtractor.get_positional_features(landmarks, img.shape)

                if positional is None:
                    continue

                position, rotation = positional
                rotation = rotations.append_and_filter(rotation)

                print position*img.shape[0]
                cv2.circle(img, tuple((position*img.shape[0]).astype(dtype=np.int32)), 2, (0, 0, 255), 2)

                geometric = utility_geometric.get_hand_crafted_geometric_features(landmarks, img.shape[0])

                result = np.concatenate((geometric, position, rotation), axis=0)
                features[idx_name][name][frame] = result

                if pose is not None:
                    rotation_vector, translation_vector = pose

                    rotation_vector = utility_positional.euler_to_rot_vec(rotation)
                    # rotation_vector = utility_positional.euler_to_rot_vec(rotations[N - 1, :])

                    # Project a 3D point (0, 0, 1000.0) onto the image plane.
                    # We use this to draw a line sticking out of the nose

                    (nose_end_point2D, _) = cv2.projectPoints(
                        np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector,
                        poseExtractor.camera_matrix, poseExtractor.dist_coeffs)

                    p1 = (int(landmarks[30, 0]), int(landmarks[30, 1]))
                    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                    cv2.line(img, p1, p2, (255, 0, 0), 2)

            cv2.imshow("", img)
            k = cv2.waitKey(1)
            if k == 2555904:  # wait for right arrow
                cv2.destroyAllWindows()
                break
            elif k == 27:  # wait for ESC
                cv2.destroyAllWindows()
                kill_all = True
                break

# cPickle.dump(features, dumpfile, -1)
