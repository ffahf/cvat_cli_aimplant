import PIL.Image
from ultralytics import YOLO
import cvat_sdk.auto_annotation as cvataa
import cvat_sdk.models as models

# Load your trained YOLOv8 Pose model
MODEL_PATH = "C:/Users/fahfi/capstone/pilot_cvat/runs/pose/train15/weights/best.pt"
_model = YOLO(MODEL_PATH)

def _yolo_to_cvat(results):
    for result in results:
        for keypoints, label in zip(result.keypoints.xy, result.boxes.cls):
            elements = [
                cvataa.keypoint(
                    label_id=i ,  # Ensure it matches the sublabel ID in `spec`
                    points=keypoint.tolist()
                )
                for i, keypoint in enumerate(keypoints)
            ]
            yield cvataa.skeleton(
                label_id=int(label.item()),  # The object label ID
                elements=elements
            )



# Function to run the model and return annotations
def detect(context, image):
    return list(_yolo_to_cvat(_model.predict(source=image, verbose=False)))