# swap_face_image.py

import cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# Load source and target images
source_path = "source.jpg"
target_path = "target.jpg"
output_path = "output.jpg"

source_img = cv2.imread(source_path)
target_img = cv2.imread(target_path)

# Face detection
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

source_faces = app.get(source_img)
target_faces = app.get(target_img)

if not source_faces or not target_faces:
    raise Exception("No faces found in one of the images")

# Use the first face
source_face = source_faces[0]
target_face = target_faces[0]

# Load the swapper model
swapper = get_model('inswapper_128.onnx', download=True, providers=['CPUExecutionProvider'])

# Swap
swapped_img = swapper.get(target_img, target_face, source_face)

# Save output
cv2.imwrite(output_path, swapped_img)
print(f"✅ Face swapped successfully: {output_path}")
