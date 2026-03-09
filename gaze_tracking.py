import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

# Use my face coordinates for computing the transformation
# of a 3D face onto the 2D facemesh landmarks from mediapipe
# x-axis : positive is in the direction of the user's left
# y-axis : positive is upward
# z-axis : positive is forward from the user's face toward the webcam
# coordinate measurements are in millimeters
FACE_3D_COORDINATES = np.array([
    [0.0, 0.0, 0.0], # nose tip is the origin
    [0.0, -95.0, -20.0], # chin
    [50.0, 40.0, -50.0], # left eye outer corner
    [-52.5, 45.0, -50.0], # right eye outer corner
    [25.0, -42.5, -25.0], # left mouth corner
    [-25.0, -40.0, -37.5], # right mouth corner
], dtype=np.float64)

# mediapipe facemesh assigns the above face features to these indices
FACEMESH_INDICES = [1, 152, 263, 33, 287, 57]

# Measuring my eye center provide an origin point for projecting the gaze vector
RIGHT_EYE_CENTER_3D = np.array([-30.0, 40.0, -35.0], dtype=np.float64)
LEFT_EYE_CENTER_3D = np.array([30.0, 40.0, -35.0], dtype=np.float64)

# This variable lets us compute how far the iris has moved fom the center of the eye
EYE_RADIUS = 15.0

def estimate_head_pose(landmarks, frame_width, frame_height):
    """
    Compute 3D head pose relative to camera. Uses Perpective-n-Point (PnP) solver from CV2.
    Because we have the measured points of facial features in 3D, we can now use the known points
    on the 2D facemesh image annotation to find a rotation and translation vector of the head pose
    relative the to camera.
    args:
        landmarks: mediapipe facemesh landmarks
        frame_width: camera frame width in pixels
        frame_height: camera frame height in pixels
    returns:
        roation_vector: represents the head's orientation where the vector direction is the axis of rotation
                        and the magnitude is the angle of rotation in radians
        translation_vector: position of the head relative to the camera
        camera_matrix: matrix representing intrinsic properties of the camera like focal length and optical center
    """

    # mediapipe represents each landmark as a ratio from 0 to 1
    # so we have to scale this by the frame dimensions to get the pixel coordinates
    face_points_2d = np.array([[landmarks.landmark[index].x * frame_width, landmarks.landmark[index].y * frame_height]
                          for index in FACEMESH_INDICES], dtype=np.float64)
    
    # approximation based on the webcam's field of view
    focal_length = frame_width

    # approximate the optical center as the image center
    # giving us the terms frame_width / 2 and frame_height / 2
    # the camera intrinsics matrix formula is:
    # [focal_length_x, 0,              optical_center_x]
    # [0,              focal_length_y, optical_center_y]
    # [0,              0,              1]
    camera_matrix = np.array([
        [focal_length, 0, frame_width / 2],
        [0, focal_length, frame_height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    # An example of disortion is the fisheye effect which most laptop cameras do not have
    # We can assume there is no distortion for our purposes
    camera_distortion_parameters = np.zeros((4,1), dtype=np.float64)

    # solve for the 3D head pose relative to the camera
    success_boolean, rotation_vector, translation_vector = cv2.solvePnP(
            FACE_3D_COORDINATES, face_points_2d, camera_matrix, camera_distortion_parameters,
            flags=cv2.SOLVEPNP_ITERATIVE 
    )

    return rotation_vector, translation_vector, camera_matrix

def estimate_iris_direction(landmarks, frame_width, frame_height, iris_center_idx, eye_inner_idx, eye_outer_idx, eye_top_idx, eye_bottom_idx, eye_center_3D):
    """
    Find an estimate of the gaze direction based on the position of the iris
    relative to the eye using mediapipe fashmesh landmarks.
    """
    
    # mediapipe facemesh iris position
    iris = landmarks.landmark[iris_center_idx]

    # convert to 2D pixel position
    iris_2d = np.array([iris.x * frame_width, iris.y * frame_height])

    # Find relevant eye landmark positions
    inner_eye = landmarks.landmark[eye_inner_idx]
    outer_eye = landmarks.landmark[eye_outer_idx]
    top_eye = landmarks.landmark[eye_top_idx]
    bottom_eye = landmarks.landmark[eye_bottom_idx]

    eye_left_x = min(inner_eye.x, outer_eye.x) * frame_width
    eye_right_x = max(inner_eye.x, outer_eye.x) * frame_width
    eye_top_y = top_eye.y * frame_height
    eye_bottom_y = bottom_eye.y * frame_height

    # average the two values to get a better approximation
    eye_center_2d = np.array([
        (eye_left_x + eye_right_x) / 2,
        (eye_top_y + eye_bottom_y) / 2
    ])

    eye_width = eye_right_x - eye_left_x
    eye_height = eye_bottom_y - eye_top_y

    # return looking forward if eyes are centered
    if eye_width < 1e-6 or eye_height < 1e-6:
        return np.array([0.0, 0.0, -1.0])
    
    # how far the iris is relative to center
    # negative offset means left of center
    # positive offset means right of center
    iris_offset_x = (iris_2d[0] - eye_center_2d[0]) / (eye_width / 2)
    iris_offset_y = (iris_2d[1] - eye_center_2d[1]) / (eye_height / 2)

    # convert 2D iris offset to 3D gaze direction
    gaze_direction = np.array([
        iris_offset_x * EYE_RADIUS,
        iris_offset_y * EYE_RADIUS,
        -EYE_RADIUS
    ], dtype=np.float64)

    # normalize the vector to ensure we have pure rotation
    # with magnitude of 1
    norm = np.linalg.norm(gaze_direction)
    if norm > 1e-6:
        gaze_direction /= norm
    
    return gaze_direction

def gaze_to_screen_point(landmarks, frame_width, frame_height,
                         screen_width_pixels, screen_height_pixels):
    """
    Use facemesh landmarks and screen dimensions to compute the head pose
    and gaze vector. Next, compute the x and y coordinates of the pixel
    being viewed on the screen.
    returns:
        screen_x: horizontal pixel position from gaze
        screen_y: vertical pixel position from gaze
    """
    # compute the head pose rotation and translation
    rotation_vector, translation_vector, camera_matrix = estimate_head_pose(landmarks,
                                                                            frame_width,
                                                                            frame_height)
    
    # rotation matrix from rotation vector
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # estimate the gaze for right and left eyes
    right_gaze = estimate_iris_direction(
        landmarks, frame_width, frame_height,
        iris_center_idx=468, eye_inner_idx=133,
        eye_outer_idx=33,
        eye_top_idx=159, eye_bottom_idx=145,
        eye_center_3D=RIGHT_EYE_CENTER_3D
    )

    left_gaze = estimate_iris_direction(
        landmarks, frame_width, frame_height,
        iris_center_idx=473, eye_inner_idx=362,
        eye_outer_idx=263,
        eye_top_idx=386, eye_bottom_idx=374,
        eye_center_3D=LEFT_EYE_CENTER_3D
    )

    # average and normalize gaze to get a better estimate
    average_gaze_local = (right_gaze + left_gaze) / 2
    average_gaze_local /= np.linalg.norm(average_gaze_local)

    # convert gaze from head frame to camera frame
    gaze_world = rotation_matrix @ average_gaze_local

    # if gaze is parallel to the screen
    # there can be a risk of division by zero
    if abs(gaze_world[2]) < 1e-6:
        # default to screen center
        return screen_width_pixels / 2, screen_height_pixels / 2
    
    # horizontal and vertical gaze ratios
    # positive gaze horizontal ratio means user is looking right on the screen
    # positive vertical ratio means the user is looking up
    gaze_horizontal_ratio = gaze_world[0] / gaze_world[2]
    gaze_vertical_ratio = gaze_world[1] / gaze_world[2]

    # account for how sensitive the screen gaze
    # location estimate is to the gaze angle
    # these parameters can be modified to ensure
    # better gaze tracking performance
    horizontal_scale_factor = 2.0
    vertical_scale_factor = 2.0

    # estimate the x and y coordinates of the gaze
    screen_x = (0.5 + gaze_horizontal_ratio * horizontal_scale_factor) * screen_width_pixels
    screen_y = (0.5 + gaze_vertical_ratio * vertical_scale_factor) * screen_height_pixels

    # ensure the boundary of the screen is the limit of the x and y pixel coordinates
    screen_x = np.clip(screen_x, 0, screen_width_pixels)
    screen_y = np.clip(screen_y, 0, screen_height_pixels)

    return float(screen_x), float(screen_y)

class GazeCalibration:
    """
    Calibrate the gaze vector by learning a mapping from
    gaze estimates to true known points on the screen.
    This method accounts for errors in the camera
    focal length approximation, camera optical center,
    3D face measurements, and all other variables with
    potential estimation error.
    """
    def __init__(self):
        # what is the gaze estimation for a given point on the screen
        self.collected_points = []
        # screen position of calibration point
        self.true_points = []

        # these are added with the fit method
        self.model_x = None
        self.model_y = None

    def add_sample(self, collected_point_x, collected_point_y,
                   true_point_x, true_point_y):
        # record collected points with their corresponding true screen locations
        self.collected_points.append([collected_point_x,collected_point_y])
        self.true_points.append([true_point_x, true_point_y])

    def fit(self):
        """
        Correct for translation, scaling, and rotation of the gaze estimates.
        The accumulated errors of gaze estimation could be caused by a slightly tilted head
        during gaze tracking or errors in estimating the intrinsic camera paramaters.
        The linear regression model accounts for these errors
        when comparing to the true known points during calibration.
        """
        self.model_x = LinearRegression().fit(self.collected_points,
                            [point[0] for point in self.true_points])
        self.model_y = LinearRegression().fit(self.collected_points,
                            [point[1] for point in self.true_points])

    def transform(self, collected_point_x, collected_point_y):
        """
        Return the corrected x and y coordinates using
        the fitted linear regression model after calibration is finished.
        """
        input_point = np.array([[collected_point_x, collected_point_y]])
        return float(self.model_x.predict(input_point)[0]), \
                float(self.model_y.predict(input_point)[0])
    
class GazeSmoothTracking:
    """
    The calibrated gaze to screen position can be jumpy as the eye moves quickly.
    This class uses a moving average to make the tracking smoother for
    a better user experience.
    """
    def __init__(self, alpha=0.3):
        # smoothing factor in the range 0 to 1
        # lower values make the gaze tracking smoother but less responsive
        # higher values make the gaze tracking more responsive but more jumpy
        self.alpha = alpha
        self.smooth_x = None
        self.smooth_y = None

    def update(self, x, y):
        if self.smooth_x is None:
            # initialize the moving average with the first frame's data
            # after calibration step
            self.smooth_x, self.smooth_y = x, y
        else:
            # update the new coordinate with the computed
            # smoothed average of previous frames
            self.smooth_x = self.alpha * x + (1 - self.alpha) * self.smooth_x
            self.smooth_y = self.alpha * y + (1 - self.alpha) * self.smooth_y

        return self.smooth_x, self.smooth_y
