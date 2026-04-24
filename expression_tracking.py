import cv2
import mediapipe as mp
from mediapipe.python.solutions import face_mesh as face_tracking
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

settings = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=settings,
                                       output_face_blendshapes=True,
                                       num_faces=1)
expression_detector = vision.FaceLandmarker.create_from_options(options)

def compute_expression(frame):
    # Process image with MediaPipe
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #convert image format to mediapipe
    mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    expression_detection = expression_detector.detect(mediapipe_image)

    smile_score = 0.
    frown_score = 0.
    neutral_score = 0.
    if expression_detection.face_blendshapes:
        frame = cv2.flip(frame, 1)  # Mirror the camera 
        for category in expression_detection.face_blendshapes[0]:
            if category.category_name == "_neutral":
                neutral_score = category.score
            if category.category_name == "mouthFrownLeft":
                frown_score += category.score
            if category.category_name == "mouthFrownRight":
                frown_score += category.score
            if category.category_name == "mouthSmileLeft":
                smile_score += category.score
            if category.category_name == "mouthSmileRight":
                smile_score += category.score

        if smile_score > 1.0:
            expression_status = "expression: smiling"
        elif frown_score > 0.005:
            expression_status = "expression: frowning"
        else:
            expression_status = "expression: neutral"
        cv2.putText(frame, expression_status, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 255, 0), 2)
        
        # reset scores for the next frame
        neutral_score = 0.
        frown_score = 0.
        smile_score = 0.