#!/usr/bin/python

import numpy as np
from math import *
from numpy import inf
import sys


def get_slope(p1, p2):
    return (p1[1] - p2[1]) / (p1[0] - p2[0])


def get_distance_between2points(p1, p2, face_height):
    return np.linalg.norm(p2-p1)/face_height


def get_angle_between2points(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def clean_cos(cos_angle):
    return min(1, max(cos_angle, -1))


def get_angle_between3points(p1, p2, p3):
    p12 = p1 - p2
    p23 = p1 - p3
    angle = atan2(p12[0] * p23[1] - p12[1] * p23[0], p12[0] * p23[0] + p12[1] * p23[1])
    return angle


def get_curvature2points(p1, p2, face_height):
    return 1 / (get_distance_between2points(p1, p2, face_height))


def get_curvature3points(p1, p2, p3):
    # sources: http: // mathworld.wolfram.com / Circle.html
    # https: // nl.mathworks.com / matlabcentral / newsreader / view_thread / 115679

    # KFROM3POINTS Calculate curvature of circle through the points

    # K = KFROM3POINTS(XS, YS)
    xs = np.array([p1[0], p2[0], p3[0]])
    xs = xs.transpose()
    ys = np.array([p1[1], p2[1], p3[1]])
    ys = ys.transpose()

    ss = np.power(xs, 2) + np.power(ys, 2)
    os = np.array([1, 1, 1])
    os = os.transpose()

    a = np.linalg.det([xs, ys, os])
    d = - np.linalg.det([ss, ys, os])
    e = np.linalg.det([ss, xs, os])
    f = - np.linalg.det([ss, xs, ys])
    if a == 0:
        a = sys.float_info.epsilon
    r = sqrt((pow(d, 2) + pow(e, 2)) / (4 * pow(a, 2)) - (f / a))  # Eq. 30

    return 1/r


brow_l_outer = 17
brow_l_inner = 21
brow_r_inner = 22
brow_r_outer = 26
eye_l_outer = 36
eye_l_inner = 39
eye_r_inner = 42
eye_r_outer = 45
mouth_l_outer = 48
mouth_l_inner = 60
mouth_r_inner = 64
mouth_r_outer = 54
mouth_t_outer = 51
mouth_b_outer = 57



def get_hand_crafted_geometric_features(image_facial_landmarks, face_height):
    # Eye aspect ratio(LR)
    p1 = np.mean(image_facial_landmarks[36:41], axis=0)
    p2 = np.mean(image_facial_landmarks[42:47], axis=0)
    eye_asp_rat = get_distance_between2points(p1, p2, face_height)

    geometric_feats = np.array([eye_asp_rat])

    # Mouth aspect ratio !!!
    p1 = image_facial_landmarks[mouth_l_outer]
    p2 = image_facial_landmarks[mouth_t_outer]
    p3 = image_facial_landmarks[mouth_r_outer]
    p4 = image_facial_landmarks[mouth_b_outer]

    dist1 = get_distance_between2points(p1, p3, face_height)
    dist2 = get_distance_between2points(p2, p4, face_height)
    if dist2 == 0:
        dist2 = sys.float_info.epsilon
    mouth_asp_rat = dist1 / dist2
    geometric_feats = np.hstack((geometric_feats, [dist1, dist2, mouth_asp_rat]))

    # Upper lip angles(LR)
    p1 = image_facial_landmarks[mouth_l_outer]
    p2 = image_facial_landmarks[mouth_t_outer]
    p3 = image_facial_landmarks[mouth_r_outer]

    u_lip_anlge = get_angle_between3points(p1, p2, p3)
    geometric_feats = np.hstack((geometric_feats, [u_lip_anlge]))

    # Nose tip - mouth corner angles(LR)

    p1 = image_facial_landmarks[33]
    p2 = image_facial_landmarks[mouth_l_outer]
    p3 = image_facial_landmarks[mouth_r_outer]

    nose_tip = get_angle_between3points(p1, p2, p3)
    geometric_feats = np.hstack((geometric_feats, [nose_tip]))

    # Lower lip angles(LR)
    p1 = np.mean([image_facial_landmarks[mouth_l_outer], image_facial_landmarks[58]], axis=0)
    p2 = np.mean([image_facial_landmarks[mouth_r_outer], image_facial_landmarks[56]], axis=0)

    l_lip_angle = get_angle_between2points(p1, p2)
    geometric_feats = np.hstack((geometric_feats, [l_lip_angle]))

    # Eyebrow slope(LR)
    p1 = np.mean([image_facial_landmarks[brow_l_outer], image_facial_landmarks[brow_l_inner]], axis=0)
    p2 = np.mean([image_facial_landmarks[brow_r_inner], image_facial_landmarks[brow_r_outer]], axis=0)

    eyebrow_slope = get_slope(p1, p2)
    geometric_feats = np.hstack((geometric_feats, [eyebrow_slope]))

    # Lower  eye angles(LR)

    p1 = np.mean([image_facial_landmarks[eye_l_outer], image_facial_landmarks[eye_l_inner],
                  image_facial_landmarks[40], image_facial_landmarks[41]], axis=0)
    p2 = np.mean([image_facial_landmarks[eye_r_inner], image_facial_landmarks[eye_r_outer],
                  image_facial_landmarks[46], image_facial_landmarks[47]], axis=0)

    l_eye_angle = get_angle_between2points(p1, p2)
    geometric_feats = np.hstack((geometric_feats, [l_eye_angle]))

    # mouth corner - mouth bottom angles
    p1 = image_facial_landmarks[mouth_l_outer]
    p2 = image_facial_landmarks[mouth_r_outer]
    p3 = image_facial_landmarks[mouth_b_outer]

    mouth_corner = get_angle_between3points(p1, p2, p3)
    geometric_feats = np.hstack((geometric_feats, [mouth_corner]))

    # upper mouth angle
    p1 = np.mean([image_facial_landmarks[mouth_l_outer],  image_facial_landmarks[50]], axis=0)
    p2 = np.mean([image_facial_landmarks[52], image_facial_landmarks[mouth_r_outer]], axis=0)

    u_mouth_angle = get_angle_between3points(p1, p2, p3)
    geometric_feats = np.hstack((geometric_feats, [u_mouth_angle]))

    # Curvature of lower - outer lips
    p1 = np.mean([image_facial_landmarks[mouth_l_outer], image_facial_landmarks[59], image_facial_landmarks[58]], axis=0)
    p2 = np.mean([image_facial_landmarks[mouth_r_outer], image_facial_landmarks[55], image_facial_landmarks[56]], axis=0)

    c_lower_outer_lips = get_curvature2points(p1, p2, face_height)
    geometric_feats = np.hstack((geometric_feats, [c_lower_outer_lips]))

    # Curvature of lower - inner lips
    p1 = np.mean([image_facial_landmarks[mouth_l_outer], image_facial_landmarks[58], image_facial_landmarks[mouth_b_outer]], axis=0)
    p2 = np.mean([image_facial_landmarks[mouth_r_outer], image_facial_landmarks[56], image_facial_landmarks[mouth_b_outer]], axis=0)

    c_lower_inner_lips = get_curvature2points(p1, p2, face_height)
    geometric_feats = np.hstack((geometric_feats, [c_lower_inner_lips]))

    # Bottom lip curvature(three points curvature!!!)
    p1 = image_facial_landmarks[mouth_l_outer]
    p2 = image_facial_landmarks[mouth_r_outer]
    p3 = image_facial_landmarks[mouth_b_outer]

    b_lip_curvature = get_curvature3points(p1, p2, p3)
    geometric_feats = np.hstack((geometric_feats, [b_lip_curvature]))

    # Mouth opening / mouth width
    p1 = image_facial_landmarks[62]
    p2 = image_facial_landmarks[66]

    mouth_opening = get_distance_between2points(p1, p2, face_height)
    geometric_feats = np.hstack((geometric_feats, [mouth_opening]))

    p1 = image_facial_landmarks[mouth_l_outer]
    p2 = image_facial_landmarks[mouth_r_outer]

    mouth_width = get_distance_between2points(p1, p2, face_height)
    geometric_feats = np.hstack((geometric_feats, [mouth_width]))
    geometric_feats = np.hstack((geometric_feats, [mouth_opening / mouth_width]))

    # Mouth up / low !!! three points distances
    p1 = image_facial_landmarks[mouth_t_outer]
    p2 = image_facial_landmarks[mouth_b_outer]
    p3 = image_facial_landmarks[62]
    p4 = image_facial_landmarks[66]

    mouth_up = get_distance_between2points(p1, p2, face_height)
    mouth_low = get_distance_between2points(p4, p3, face_height)
    mouth_inner = get_distance_between2points(p1, p3, face_height)
    mouth_outer = get_distance_between2points(p2, p4, face_height)
    geometric_feats = np.hstack((geometric_feats, [mouth_up, mouth_low, mouth_inner, mouth_outer]))

    # Eye - middle eyebrow distance(LR)
    p1 = np.mean([image_facial_landmarks[19], image_facial_landmarks[eye_l_outer], image_facial_landmarks[eye_l_inner]], axis=0)
    p2 = np.mean([image_facial_landmarks[24], image_facial_landmarks[eye_r_inner], image_facial_landmarks[eye_r_outer]], axis=0)

    middle_eyebrow_dist = get_distance_between2points(p1, p2, face_height)
    geometric_feats = np.hstack((geometric_feats, [middle_eyebrow_dist]))

    # Eye - inner eyebrow distance(LR)
    p1 = np.mean([image_facial_landmarks[brow_l_inner], image_facial_landmarks[eye_l_outer], image_facial_landmarks[eye_l_inner]], axis=0)
    p2 = np.mean([image_facial_landmarks[brow_r_inner],  image_facial_landmarks[eye_r_inner], image_facial_landmarks[eye_r_outer]], axis=0)

    inner_eyebrow_dist = get_distance_between2points(p1, p2, face_height)
    geometric_feats = np.hstack((geometric_feats, [inner_eyebrow_dist]))
    # Inner eye - eyebrow center(LR)

    p1 = np.mean([image_facial_landmarks[19], image_facial_landmarks[eye_l_inner]], axis=0)
    p2 = np.mean([image_facial_landmarks[24], image_facial_landmarks[eye_r_inner]], axis=0)

    inner_eye_eyebrow_center = get_distance_between2points(p1, p2, face_height)
    geometric_feats = np.hstack((geometric_feats, [inner_eye_eyebrow_center]))

    # Inner eye - mouth top distance
    p1 = image_facial_landmarks[eye_l_inner]
    p2 = image_facial_landmarks[eye_r_inner]
    p3 = image_facial_landmarks[mouth_t_outer]

    eye_center = np.mean([p1, p2], axis=0)

    inner_eye_mouth_dist = get_distance_between2points(eye_center, p3, face_height)
    left_eye_mouth_dist = get_distance_between2points(p1, p3, face_height)
    right_eye_mout_dist = get_distance_between2points(p2, p3, face_height)

    geometric_feats = np.hstack((geometric_feats, [inner_eye_mouth_dist, left_eye_mouth_dist, right_eye_mout_dist]))

    # Mouth width
    p1 = image_facial_landmarks[mouth_l_outer]
    p2 = image_facial_landmarks[mouth_r_outer]

    mouth_width = get_distance_between2points(p1, p2, face_height)
    geometric_feats = np.hstack((geometric_feats, [mouth_width]))

    # Mouth height
    p1 = image_facial_landmarks[mouth_t_outer]
    p2 = image_facial_landmarks[mouth_b_outer]

    mouth_height = get_distance_between2points(p1, p2, face_height)
    geometric_feats = np.hstack((geometric_feats, [mouth_height]))

    # Upper mouth  height
    p1 = np.mean([image_facial_landmarks[mouth_l_outer], image_facial_landmarks[mouth_r_outer]], axis=0)
    p2 = image_facial_landmarks[mouth_t_outer]

    u_mouth_height = get_distance_between2points(p1, p2, face_height)
    geometric_feats = np.hstack((geometric_feats, [u_mouth_height]))

    # Lower mouth height
    p1 = np.mean([image_facial_landmarks[mouth_l_outer], image_facial_landmarks[mouth_r_outer]], axis=0)
    p2 = image_facial_landmarks[mouth_b_outer]

    l_mouth_height = get_distance_between2points(p1, p2, face_height)
    geometric_feats = np.hstack((geometric_feats, [l_mouth_height]))
    # pdb.set_trace()

    geometric_feats[geometric_feats == -inf] = 0
    geometric_feats = np.nan_to_num(geometric_feats)
    return geometric_feats

# facial landmark mapping from intraface to dlib
# based on this link: https://cdn-images-1.medium.com/max/800/1*AbEg31EgkbXSQehuNJBlWg.png
# image_facial_landmarks[brow_l_inner] / 5
# image_facial_landmarks[brow_r_inner] / 6
# image_facial_landmarks[19] / 3
# image_facial_landmarks[24] / 8
# image_facial_landmarks[33] / 17
# image_facial_landmarks[mouth_l_outer] / 32
# image_facial_landmarks[mouth_t_outer] / 35
# image_facial_landmarks[mouth_r_outer] / 38
# image_facial_landmarks[mouth_b_outer] / 41
# image_facial_landmarks[58] / 42
# image_facial_landmarks[56] / 40
# image_facial_landmarks[brow_l_outer] / 1
# image_facial_landmarks[brow_l_inner] / 5
# image_facial_landmarks[brow_r_inner] / 6
# image_facial_landmarks[brow_r_outer] / 10
# image_facial_landmarks[eye_l_outer] / 20
# image_facial_landmarks[eye_l_inner] / 23
# image_facial_landmarks[40] / 24
# image_facial_landmarks[41] / 25
# image_facial_landmarks[eye_r_inner] / 26
# image_facial_landmarks[eye_r_outer] / 29
# image_facial_landmarks[46] / 30
# image_facial_landmarks[47] / 31
# image_facial_landmarks[50] / 34
# image_facial_landmarks[52] / 36
# image_facial_landmarks[59] / 43
# image_facial_landmarks[55] / 39
# image_facial_landmarks[62] / 45
# image_facial_landmarks[66] / 48
