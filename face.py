import cv2
from insightface.app import FaceAnalysis

face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

img = cv2.imread("your_face.jpg")
faces = face_analyzer.get(img)

print(f"✅ Faces detected: {len(faces)}")
if faces:
    for face in faces:
        box = face.bbox.astype(int)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imshow("Detected Face", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
