import numpy as np
import cv2
import math


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rot_vec_to_euler(r):
    # Rotate around x axis by 180 degrees to have [0, 0, 0] when facing forward
    R = np.dot(np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]]),
               np.array(cv2.Rodrigues(r)[0]))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


# Calculates Rotation Matrix given euler angles.
def euler_to_rot_vec(theta):
    r_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    r_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    r_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    return np.array(cv2.Rodrigues(np.dot(np.array([[1, 0, 0],
                                                   [0, -1, 0],
                                                   [0, 0, -1]]),
                                         np.dot(r_z, np.dot(r_y, r_x))))[0])


class PoseExtractor:
    def __init__(self):
        self.image_points = np.array([30, 29, 28, 27, 33, 32, 34, 31, 35,
                                      36, 45, 39, 42,
                                      21, 22, 20, 23, 19, 24, 18, 25
                                      ], dtype=np.intp)

        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, 0.40412, -0.35702),  # Nose 1
            (0.0, 0.87034, -0.65485),  # Nose 2
            (0, 1.33462, -0.92843),  # Nose 3
            (0, -0.63441, -0.65887),  # Under Nose #0
            (0, 0, 0),  # Under Nose #1, L
            (0.25466, -0.59679, -0.80215),  # Under Nose #1, R
            (0, 0, 0),  # Under Nose #2, L
            (0.49277, -0.56169, -0.96709),  # Under Nose #2, R
            (0, 0, 0),  # Left eye outer corner
            (1.60745, 1.21855, -1.9585),  # Right eye outer corner
            (0, 0, 0),  # Left eye inner corner
            (0.53823, 1.15389, -1.37273),  # Right eye inner corner
            (0, 0, 0),  # Eyebrow #0, L
            (0.34309, 1.67208, -0.96486),  # Eyebrow #0, R
            (0, 0, 0),  # Eyebrow #1, L
            (0.65806, 1.85405, -1.04975),  # Eyebrow #1, R
            (0, 0, 0),  # Eyebrow #2, L
            (0.96421, 1.95277, -1.23015),  # Eyebrow #2, R
            (0, 0, 0),  # Eyebrow #3, L
            (1.32075, 1.95305, -1.48482)  # Eyebrow #3, R
        ])

        for i in range(5, self.model_points.shape[0], 2):
            self.model_points[i, 0] = -self.model_points[i + 1, 0]
            self.model_points[i, 1:3] = self.model_points[i + 1, 1:3]

        self.camera_matrix = None  # Hack so camera matrix can be used for printing later

        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        self.rvec = None
        self.tvec = None

    def get_head_rotation(self, landmarks, img_size):
        # Camera internals
        focal_length = img_size[1]
        center = (img_size[1] / 2, img_size[0] / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        if self.rvec is None:
            (success, self.rvec, self.tvec) = cv2.solvePnP(
                self.model_points, landmarks[self.image_points[:, np.newaxis], :],
                self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
        else:
            (success, self.rvec, self.tvec) = cv2.solvePnP(
                self.model_points, landmarks[self.image_points[:, np.newaxis], :],
                self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_EPNP,
                rvec=self.rvec, tvec=self.tvec, useExtrinsicGuess=True)

        return success

    def get_positional_features(self, landmarks, img_size):
        rotation_success = self.get_head_rotation(landmarks, img_size)
        if not rotation_success:
            return None

        return self.tvec, rot_vec_to_euler(self.rvec)


def get_position_by_average(landmarks, img_size):
    position = np.mean(landmarks, axis=0)
    size = 2 * np.mean(np.linalg.norm((landmarks - position), axis=1, ord=2))

    return np.append(position / img_size[0], size / img_size[0])
