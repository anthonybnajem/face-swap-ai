import os
import subprocess
import cv2
import time
import numpy as np
import tempfile
import requests
from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper
from numpy.linalg import norm
from urllib.parse import urlparse

def swap_faces_in_video(
    video_path: str,
    source_image_path: str,  # Can be a local path or URL
    output_video_path: str,
    final_output_path: str,
    target_indices: list[int],
    detected_dir: str = "detected_faces",
    n_frames_scan: int = 500,
    threshold: float = 0.4
):
    # --- Setup ---
    os.makedirs(detected_dir, exist_ok=True)

    print("🔧 Initializing models...")
    face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    swapper = INSwapper(model_file=os.path.expanduser("~/.insightface/models/inswapper_128/inswapper_128.onnx"))

    # --- Load source face (support URL) ---
    print("📥 Loading source face...")
    if urlparse(source_image_path).scheme in ('http', 'https'):
        try:
            response = requests.get(source_image_path, timeout=10)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name
            source_img = cv2.imread(tmp_path)
            os.unlink(tmp_path)  # Clean up after loading
        except Exception as e:
            raise RuntimeError(f"❌ Failed to download or read image from URL: {e}")
    else:
        source_img = cv2.imread(source_image_path)

    if source_img is None:
        raise RuntimeError(f"❌ Could not load image from `{source_image_path}`.")

    source_faces = face_analyzer.get(source_img)
    if not source_faces:
        raise RuntimeError(f"❌ No face found in source image `{source_image_path}`.")
    source_face = source_faces[0]

    # --- Cosine Distance ---
    def cosine_distance(v1, v2):
        return 1 - np.dot(v1, v2) / (norm(v1) * norm(v2) + 1e-5)

    # --- Step 1: Detect unique faces ---
    print(f"\n🔍 Scanning first {n_frames_scan} frames from `{video_path}` for unique faces...")
    cap = cv2.VideoCapture(video_path)
    collected_faces = []
    face_counter = 0

    for _ in range(n_frames_scan):
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_analyzer.get(frame)
        for face in faces:
            emb = face.normed_embedding
            if emb is None:
                continue

            is_new = all(
                cosine_distance(emb, old.normed_embedding) >= threshold
                for old in collected_faces
            )

            if is_new:
                collected_faces.append(face)
                box = face.bbox.astype(int)
                face_crop = frame[box[1]:box[3], box[0]:box[2]]
                save_path = os.path.join(detected_dir, f"{face_counter}.jpg")
                cv2.imwrite(save_path, face_crop)
                print(f"📸 Saved face {face_counter} to `{save_path}`")
                face_counter += 1

    cap.release()
    if not collected_faces:
        raise RuntimeError("❌ No unique faces detected in the video.")

    # --- Step 2: Swap Faces ---
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    print("\n🔁 Starting face swap on full video...")

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
                    if dist < threshold:
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
    print(f"\n✅ Face swap complete! Saved to `{output_video_path}`")

    # --- Step 3: Add original audio ---
    print("🔊 Merging original audio...")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", output_video_path,
        "-i", video_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-c:a", "copy",
        "-movflags", "+faststart",
        final_output_path
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"🎉 Final video with audio saved to `{final_output_path}`")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FFmpeg audio merge failed: {e}")
