#!/usr/bin/python

import cv2
import numpy as np
from numpy import inf


def dist(p1, p2):
    return np.linalg.norm(p2 - p1)


def angle_2(p1, p2):
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])


def angle_3(p1, p2, p3):
    p12 = p1 - p2
    p13 = p1 - p3
    return np.arccos(np.dot(p12, p13) / (np.linalg.norm(p12) * np.linalg.norm(p13)))


def angle(*args):
    if len(args) == 2:
        return angle_2(*args)
    elif len(args) == 3:
        return angle_3(*args)
    assert False


eye_l_start = 36
eye_l_end = 42
eye_r_start = 42
eye_r_end = 48


# FACS
# 1  Inner Brow Raiser
# 2  Outer Brow Raiser
# 4  Brow Lowerer
# 5  Upper Lid Raiser
# 6  Cheek Raiser
# 7  Lid Tightener
# 8  Lips Toward Each Other
# 9  Nose Wrinkler
# 10 Upper Lip Raiser
# 11 Nasolabial Furrow Deepener
# 12 Lip Corner Puller
# 13 Cheek puffer
# 14 Dimpler
# 15 Lip Corner Depressor
# 16 Lower Lip Depressor
# 17 Chin Raiser
# 18 Lip Puckerer
# 20 Lip Stretcher
# 22 Lip Funneler
# 23 Lip Tightner
# 24 Lip Pressor
# 25 Lips Part
# 26 Jaw Drop
# 27 Mouth Stretch
# 28 Lip suck
# 38 Nostril Dilator
# 39 Nostril Compressor
# 41 Lid Droop
# 42 Slit
# 43 Eyes Closed
# 44 Squint
# 45 Blink
# 46 Wink

def get_hand_crafted_geometric_features(l):
    # Transform landmarks so that eyes and mouth are on fixed points
    # This keeps slope calculations invariant to head roll
    # while also mitigating the effects of yaw and pitch changes
    eye_l = np.mean(l[eye_l_start:eye_l_end, :], axis=0)
    eye_r = np.mean(l[eye_r_start:eye_r_end, :], axis=0)
    upper_lip = np.mean(l[48:55, :], axis=0)
    align_centroids = np.vstack((eye_l, eye_r, upper_lip))
    target_points = np.float32([[0, 1],
                                [1, 1],
                                [0.5, 0]])
    shape_before = l.shape
    l = np.dot(
        cv2.getAffineTransform(align_centroids, target_points),
        np.vstack((l.T, np.ones([1, 68])))).transpose()[:, :2]
    assert (l.shape == shape_before)
    assert (len(l[0]) == 2)
    assert (l[:2].shape == (2, 2))

    centroid = np.mean(l, axis=0)

    # recalculate because l was transformed
    eye_l = np.mean(l[eye_l_start:eye_l_end, :], axis=0)
    eye_r = np.mean(l[eye_r_start:eye_r_end, :], axis=0)

    feats = []
    # 1  Inner Brow Raiser
    # 4  Brow Lowerer
    feats += [dist(eye_l, np.mean(l[20:22], axis=0)),  # 0
              dist(eye_r, np.mean(l[22:24], axis=0))]  # 1

    # 2  Outer Brow Raiser
    # 4  Brow Lowerer
    feats += [dist(eye_l, np.mean(l[17:19], axis=0)),  # 2
              dist(eye_r, np.mean(l[25:27], axis=0))]  # 3

    # 4 Brow Lowerer
    feats.append(dist(np.mean(l[20:22]), np.mean(l[17:19])))  # 4

    nose_t = l[27]
    lid_t_l = np.mean(l[37:39], axis=0)
    lid_t_r = np.mean(l[43:45], axis=0)
    lid_b_l = np.mean(l[40:42], axis=0)
    lid_b_r = np.mean(l[46:48], axis=0)

    # 5 Upper Lid Raiser
    # 7 Lid tightener
    feats += [dist(lid_t_l, lid_b_l),  # 5
              dist(lid_t_r, lid_b_r)]  # 6

    mouth_l = np.mean(l[(48, 60), :], axis=0)
    mouth_r = np.mean(l[(54, 64), :], axis=0)
    mouth_t = np.mean(l[50:53], axis=0)
    mouth_b = np.mean(l[56:59], axis=0)

    # 12 Lip Corner Puller
    feats += [angle(mouth_t, mouth_l, mouth_r),  # 7
              angle(mouth_b, mouth_l, mouth_r)]  # 8

    # 20 Lip Stretcher
    feats.append(dist(mouth_l, mouth_r))  # 9

    # 25 Lips Part
    # 26 Jaw Drop
    feats.append(dist(mouth_t, mouth_b))  # 10

    mouth = np.mean(l[48:68, :])
    chin = np.mean(l[7:10, :])

    # 26 Jaw Drop
    feats.append(dist(mouth, chin)) #11

    geometric_feats = np.array(feats, dtype=np.float32)
    geometric_feats[geometric_feats == -inf] = 0

    return np.nan_to_num(geometric_feats)
