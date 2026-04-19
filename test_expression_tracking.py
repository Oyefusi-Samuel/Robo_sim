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
cap = cv2.VideoCapture(0)  # Connect to webcam

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Process image with MediaPipe
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #convert image format to mediapipe
        mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        expression_detection = expression_detector.detect(mediapipe_image)
        height, width = frame.shape[:2]

        smile_score = 0.
        frown_score = 0.
        neutral_score = 0.
        categories_to_process = ["_neutral", "mouthFrownLeft", "mouthFrownRight", "mouthSmileLeft", "mouthSmileRight"]
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
            neutral_status = f"neutral score: {neutral_score}"
            smile_status = f"smile score: {smile_score}"
            frown_status = f"frown score: {frown_score}"
            cv2.putText(frame, neutral_status, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, smile_status, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, frown_status, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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

        cv2.namedWindow("Test Expression Tracking", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Test Expression Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Test Expression Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    # Safe shutdown of camera
    cap.release()
    cv2.destroyAllWindows()