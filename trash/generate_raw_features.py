import cPickle
import csv
import math

import cv2
import numpy as np

import expressivity
import face_extractor
import filter
from trash import utility_positional


def draw_face(image, pts, color, thickness=1):
    open_ranges = [[0, 17], [17, 22], [22, 27], [27, 31], [31, 36]]
    closed_ranges = [[36, 42], [42, 48], [48, 60], [60, 68]]
    closed = False
    for k_range in (open_ranges, closed_ranges):
        for i_range in k_range:
            cv2.polylines(image, [np.array(pts[i_range[0]:i_range[1], :], np.int32).reshape((-1, 1, 2))], closed, color,
                          thickness)
        closed = True


def draw_coordinates(image, pt, rot, color, thickness=1):
    h = image.shape[0]
    w = image.shape[1]
    cv2.line(image, (0, h / 2), (w, h / 2), color, thickness)
    cv2.line(image, (w / 2, 0), (w / 2, h), color, thickness)

    p1 = (int(pt[0] * (w / 2)), int((1 - pt[2]) * (h / 2)))
    cv2.circle(image[:int(h / 2), :int(w / 2)], p1, 2, color, thickness)
    # p2 = np.multiply(np.array(p1, dtype=np.float32), np.array((math.cos(rot[1]), math.sin(rot[1])))) * 10
    # cv2.line(image[:int(h / 2), :int(w / 2)], p1, (int(p2[0]), int(p2[1])), color, 1)

    p1 = (int(pt[0] * (w / 2)), int(pt[1] * (h / 2)))
    cv2.circle(image[int(h / 2):, :int(w / 2)], p1, 2, color, thickness)
    # p2 = np.multiply(np.array(p1, dtype=np.float32), np.array((math.cos(rot[2]), math.sin(rot[2])))) * 10
    # cv2.line(image[int(h / 2):, :int(w / 2)], p1, (int(p2[0]), int(p2[1])), color, 1)

    p1 = (int(pt[2] * (w / 2)), int(pt[1] * (h / 2)))
    cv2.circle(image[int(h / 2):, int(w / 2):], p1, 2, color, thickness)
    # p2 = np.multiply(np.array(p1, dtype=np.float32), np.array((math.cos(rot[0]), math.sin(rot[0])))) * 10
    # cv2.line(image[int(h / 2):, int(w / 2):], p1, (int(p2[0]), int(p2[1])), color, 1)


loadfile = open('D:/Master Thesis/Dataset/landmarks.pkl', 'rb')
dumpfile = open('D:/Master Thesis/Dataset/pos_features.csv_tmptmp', 'wb')

landmarks = cPickle.load(loadfile)

extractor = face_extractor.LandmarkExtractor(10)
poseExtractor = utility_positional.poseExtractor()

# trans_filter = filter.ButterFilterArray(n=3, n_filter=1, cutoff=0.5)

# min_trans = np.array([-8.07856305, -3.9031677, 10.83461951])
# max_trans = np.array([13.52021622, 6.09885788, 43.91670756])
# fac_trans = max_trans - min_trans

writer = csv.writer(dumpfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

writer.writerow(expressivity.description(3, 'pos_') + ['file', 'frame'])

# cap = cv2.VideoCapture(0)

# while True:
#             ret, cap_img = cap.read()
#             if not ret:
#                 break
#
#             shape = cap_img.shape[0:2]
#
#             landmarks = extractor.track_landmarks(cap_img)
#             img = np.zeros((shape[0], shape[1] + shape[0], 3))

T = 20
pos_buffer = filter.NpCircularArray(n=3, length=T)
next_frame = 0

for person, videos in landmarks.iteritems():
    # if person != 'rob':
    #     continue
    for video, data in sorted(videos.iteritems()):
        next_frame = 0
        continuous = 0
        meta = data['meta']
        num_frames = meta['num_frames']
        fps = meta['fps']
        if math.isnan(fps):
            fps = 10
        shape = np.array([meta['height'], meta['width']], dtype=np.int32)

        print(video)
        # img = np.zeros((shape[0], shape[1] + shape[0], 3))

        for frame, landmarks in sorted(data.iteritems()):
            if frame == 'meta':
                continue

            if landmarks is None:
                continue

            # pose = poseExtractor.get_positional_features(landmarks, shape)

            # img.fill(0)
            # if pose:
            # trans_vec, rotation = pose

            # trans_vec = trans_filter.append_and_filter(np.reshape(trans_vec, 3))
            # draw_coordinates(img[:, shape[1]:], np.divide(np.reshape(trans_vec, 3) - min_trans, fac_trans),
            #                  rotation, (255, 255, 0), 1)

            if next_frame == frame:
                continuous += 1
            next_frame = (frame + 1) % num_frames

            position = utility_positional.get_position_by_average(landmarks, shape)
            pos_buffer.append(utility_positional.get_position_by_average(landmarks, shape))

            if continuous < T:
                continue

            pos_features = expressivity.analyze(pos_buffer.get())

            writer.writerow(pos_features.tolist() + [video, frame])

            # pos = (position[0], position[1], size*5)
            # draw_coordinates(img[:, shape[1]:], pos,
            #                  pos, (255, 255, 0), 1)
            #
            # draw_face(img[:, :shape[1]], landmarks, (0, 255, 0))
            # warped = extractor.warp_landmarks(landmarks) * min(img.shape[0:2])
            # draw_face(img[:, :shape[1]], warped, (0, 255, 255))

            # cv2.imshow("", img)
            # k = cv2.waitKey(50)
            # if k == 2555904:  # wait for right arrow
            #     cv2.destroyAllWindows()
            #     break
            # elif k == 27:  # wait for ESC
            #     cv2.destroyAllWindows()
            #     kill_all = True
            #     break
