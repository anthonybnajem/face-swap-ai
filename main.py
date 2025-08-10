from dotenv import load_dotenv
load_dotenv()

import os
import cv2
import uuid
import time
import boto3
import requests
import numpy as np
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper
from numpy.linalg import norm
from typing import List, Optional

app = FastAPI()

# AWS S3 Config
AWS_REGION = "eu-north-1"
S3_BUCKET = "vml-face-swap"

s3 = boto3.client("s3",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

def upload_file_to_s3(local_path, s3_path, content_type):
    s3.upload_file(local_path, S3_BUCKET, s3_path, ExtraArgs={
        "ACL": "public-read",
        "ContentType": content_type
    })
    return f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_path}"

# Face model
face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
swapper = INSwapper(model_file=os.path.expanduser("~/.insightface/models/inswapper_128/inswapper_128.onnx"))

# Utils
def cosine_distance(v1, v2):
    return 1 - np.dot(v1, v2) / (norm(v1) * norm(v2) + 1e-5)

def download_image_from_url(url: str, save_path: str):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
    else:
        raise RuntimeError(f"❌ Failed to download image: {url}")

# Request Models
class SwapItem(BaseModel):
    target_indices: str
    source_image: str

class SwapRequest(BaseModel):
    video_name: str
    webhook_url: Optional[str] = None
    swap: List[SwapItem]

class DetectFacesRequest(BaseModel):
    video_name: str

@app.post("/swap")
async def swap_faces(req: SwapRequest):
    uid = str(uuid.uuid4())
    video_path = f"videos/{req.video_name}.mp4"
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video {req.video_name}.mp4 not found.")

    os.makedirs("source_images", exist_ok=True)
    os.makedirs("detected_faces", exist_ok=True)
    os.makedirs("output_videos", exist_ok=True)
    detected_dir = f"detected_faces/{req.video_name}"
    output_raw = f"output_videos/{uid}_raw.mp4"
    final_output = f"output_videos/{uid}_final.mp4"
    os.makedirs(detected_dir, exist_ok=True)

    swap_item = req.swap[0]
    source_img_path = f"source_images/{req.video_name}.jpg"
    download_image_from_url(swap_item.source_image, source_img_path)
    source_img = cv2.imread(source_img_path)
    source_faces = face_analyzer.get(source_img)
    if not source_faces:
        raise HTTPException(status_code=400, detail="❌ No face found in source image.")
    source_face = source_faces[0]

    cap = cv2.VideoCapture(video_path)
    unique_faces = []
    threshold = 0.4
    N_FRAMES_SCAN = 500
    face_counter = 0

    for _ in range(N_FRAMES_SCAN):
        ret, frame = cap.read()
        if not ret:
            break
        faces = face_analyzer.get(frame)
        for face in faces:
            emb = face.normed_embedding
            if emb is None:
                continue
            is_new = all(cosine_distance(emb, f.normed_embedding) >= threshold for f in unique_faces)
            if is_new:
                unique_faces.append(face)
                box = face.bbox.astype(int)
                crop = frame[box[1]:box[3], box[0]:box[2]]
                local_img_path = os.path.join(detected_dir, f"{face_counter}.jpg")
                cv2.imwrite(local_img_path, crop)
                s3_path = f"detected_faces/{req.video_name}/{face_counter}.jpg"
                upload_file_to_s3(local_img_path, s3_path, "image/jpeg")
                face_counter += 1
    cap.release()

    if not unique_faces:
        raise HTTPException(status_code=400, detail="❌ No unique faces detected.")

    target_ids = [int(i.strip()) for i in swap_item.target_indices.split(",") if i.strip().isdigit()]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    idx, swapped_count = 0, 0
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
            for i, saved_face in enumerate(unique_faces):
                if i in target_ids and cosine_distance(emb, saved_face.normed_embedding) < threshold:
                    frame = swapper.get(frame, face, source_face, paste_back=True)
                    swapped_count += 1
                    break
        out.write(frame)
        idx += 1
        if idx % 10 == 0 or idx == total_frames:
            elapsed = time.time() - start_time
            print(f"[{(idx/total_frames)*100:.2f}%] Frame {idx}/{total_frames} | ETA: {(total_frames - idx) / (idx / elapsed):.1f}s")
    cap.release()
    out.release()

    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", output_raw,
            "-i", video_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "libx264",
            "-c:a", "copy",
            "-movflags", "+faststart",
            final_output
        ], check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg merge failed: {str(e)}")

    s3_video_path = f"output_videos/{uid}_final.mp4"
    video_url = upload_file_to_s3(final_output, s3_video_path, "video/mp4")

    if req.webhook_url:
        try:
            requests.post(req.webhook_url, json={"video_url": video_url})
        except Exception as e:
            print("⚠️ Webhook failed:", e)

    return {
        "message": "✅ Face swap complete",
        "video_url": video_url,
        "swapped_frames": swapped_count
    }

@app.post("/detect-faces-in-video")
async def detect_faces_in_video(req: DetectFacesRequest):
    video_path = f"videos/{req.video_name}.mp4"
    detected_dir = f"detected_faces/{req.video_name}"

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video {req.video_name}.mp4 not found.")

    face_urls = []
    if not os.path.exists(detected_dir) or not os.listdir(detected_dir):
        os.makedirs(detected_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        unique_faces = []
        threshold = 0.4
        N_FRAMES_SCAN = 500
        face_counter = 0

        for _ in range(N_FRAMES_SCAN):
            ret, frame = cap.read()
            if not ret:
                break
            faces = face_analyzer.get(frame)
            for face in faces:
                emb = face.normed_embedding
                if emb is None:
                    continue
                is_new = all(cosine_distance(emb, f.normed_embedding) >= threshold for f in unique_faces)
                if is_new:
                    unique_faces.append(face)
                    box = face.bbox.astype(int)
                    crop = frame[box[1]:box[3], box[0]:box[2]]
                    local_img_path = os.path.join(detected_dir, f"{face_counter}.jpg")
                    cv2.imwrite(local_img_path, crop)
                    s3_path = f"detected_faces/{req.video_name}/{face_counter}.jpg"
                    url = upload_file_to_s3(local_img_path, s3_path, "image/jpeg")
                    face_urls.append(url)
                    face_counter += 1
        cap.release()
        if not unique_faces:
            raise HTTPException(status_code=400, detail="❌ No unique faces detected.")
    else:
        # If already exists, list from S3
        for fname in sorted(os.listdir(detected_dir)):
            if fname.endswith(".jpg"):
                face_urls.append(f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/detected_faces/{req.video_name}/{fname}")

    return {
        "message": f"✅ {len(face_urls)} face(s) detected",
        "face_images": face_urls
    }
