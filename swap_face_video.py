import os
import subprocess
import cv2
import time
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper
from numpy.linalg import norm

# --- Config ---
VIDEO_PATH = "target_video_short_2.mp4"
SOURCE_IMAGE = "your_face.jpg"
OUTPUT_VIDEO = "output.mp4"
FINAL_OUTPUT = "output_with_audio.mp4"
DETECTED_DIR = "detected_faces"
N_FRAMES_SCAN = 500
EMBEDDING_SIMILARITY_THRESHOLD = 0.4  # Adjust to control uniqueness

# --- Setup ---
face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
swapper = INSwapper(model_file=os.path.expanduser("~/.insightface/models/inswapper_128/inswapper_128.onnx"))

# --- Load source face ---
source_img = cv2.imread(SOURCE_IMAGE)
source_faces = face_analyzer.get(source_img)
if not source_faces:
    raise RuntimeError("❌ No face found in your_face.jpg.")
source_face = source_faces[0]

# --- Create output dir ---
os.makedirs(DETECTED_DIR, exist_ok=True)

# --- Helper: Cosine Distance ---
def cosine_distance(v1, v2):
    return 1 - np.dot(v1, v2) / (norm(v1) * norm(v2) + 1e-5)

# --- Scan initial frames to collect unique faces ---
cap = cv2.VideoCapture(VIDEO_PATH)
collected_faces = []
face_counter = 0

print(f"\n🔍 Scanning first {N_FRAMES_SCAN} frames for unique faces...")

for _ in range(N_FRAMES_SCAN):
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_analyzer.get(frame)
    for face in faces:
        emb = face.normed_embedding
        if emb is None:
            continue

        is_new = True
        for old_face in collected_faces:
            old_emb = old_face.normed_embedding
            dist = cosine_distance(emb, old_emb)
            if dist < EMBEDDING_SIMILARITY_THRESHOLD:
                is_new = False
                break

        if is_new:
            collected_faces.append(face)
            box = face.bbox.astype(int)
            face_crop = frame[box[1]:box[3], box[0]:box[2]]
            save_path = os.path.join(DETECTED_DIR, f"{face_counter}.jpg")
            cv2.imwrite(save_path, face_crop)
            print(f"📸 Saved face {face_counter} to {save_path}")
            face_counter += 1

cap.release()

if not collected_faces:
    raise RuntimeError("❌ No unique faces detected in the video.")

# --- Ask user for which face(s) to swap ---
print(f"\n🧠 Detected {len(collected_faces)} unique faces.")
print(f"👉 Check the folder `{DETECTED_DIR}` to view them.")
user_input = input("Enter face indices to swap with your_face.jpg (comma-separated): ")
target_indices = [int(i.strip()) for i in user_input.split(",") if i.strip().isdigit()]

# --- Prepare full video for swap ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

print("\n🔁 Starting full face swap process...")
frame_idx = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_analyzer.get(frame)
    for face in faces:
        emb = face.normed_embedding
        if emb is None:
            continue

        for idx, saved_face in enumerate(collected_faces):
            if idx in target_indices:
                dist = cosine_distance(emb, saved_face.normed_embedding)
                if dist < EMBEDDING_SIMILARITY_THRESHOLD:
                    frame = swapper.get(frame, face, source_face, paste_back=True)
                    break

    out.write(frame)
    frame_idx += 1

    if frame_idx % 10 == 0 or frame_idx == total_frames:
        elapsed = time.time() - start_time
        percent = (frame_idx / total_frames) * 100
        fps_now = frame_idx / elapsed if elapsed else 0
        eta = (total_frames - frame_idx) / fps_now if fps_now else 0
        print(f"[{percent:.2f}%] {frame_idx}/{total_frames} frames | ETA: {eta:.1f}s")

cap.release()
out.release()


print(f"\n✅ Done! Output saved to `{OUTPUT_VIDEO}`")


ffmpeg_cmd = [
    "ffmpeg",
    "-y",  # Overwrite if exists
    "-i", OUTPUT_VIDEO,
    "-i", VIDEO_PATH,
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-c:v", "libx264",
    "-c:a", "copy",
    "-movflags", "+faststart",
    FINAL_OUTPUT
]

try:
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"\n✅ Final video with audio saved as `{FINAL_OUTPUT}`")
except subprocess.CalledProcessError as e:
    print(f"\n❌ FFmpeg audio merge failed: {e}")