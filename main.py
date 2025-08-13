# main.py
from dotenv import load_dotenv
load_dotenv()

import os
import cv2
import uuid
import time
import boto3
import json
import requests
import subprocess
import numpy as np
from numpy.linalg import norm
from typing import List, Optional, Dict, Any
from collections import deque

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel

from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper

app = FastAPI(title="Face Swap API", version="1.1.0")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Face Swap API", version="1.1.0")

# Allow your frontend origins
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://next-vml-face-swap.vercel.app",  # prod UI
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,          # set True only if you use cookies/auth
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── In-memory job store (single-process). Use Redis/Celery for production. ──
# job: {status, progress, result, error, eta_seconds, logs}
JOBS: Dict[str, Dict[str, Any]] = {}

# ── AWS S3 ───────────────────────────────────────────────────────────────────
AWS_REGION = "eu-north-1"
S3_BUCKET = "vml-face-swap"
s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

def upload_file_to_s3(local_path: str, s3_path: str, content_type: str) -> str:
    s3.upload_file(
        local_path,
        S3_BUCKET,
        s3_path,
        ExtraArgs={"ACL": "public-read", "ContentType": content_type},
    )
    return f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_path}"

# ── InsightFace models (loaded once) ─────────────────────────────────────────
face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
swapper = INSwapper(
    model_file=os.path.expanduser("~/.insightface/models/inswapper_128/inswapper_128.onnx")
)

# ── Helpers ──────────────────────────────────────────────────────────────────
def cosine_distance(v1, v2):
    return 1 - np.dot(v1, v2) / (norm(v1) * norm(v2) + 1e-5)

def download_image_from_url(url: str, save_path: str):
    r = requests.get(url, stream=True, timeout=30)
    if r.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        raise RuntimeError(f"Failed to download image: {url} (status {r.status_code})")

def safe_update(job_id: str, **kwargs):
    if job_id in JOBS:
        JOBS[job_id].update(**kwargs)

def job_log(job_id: str, msg: str):
    """Append a log line to the job's ring buffer and also print to console."""
    entry = f"{time.strftime('%H:%M:%S')} | {msg}"
    print(entry)
    if job_id in JOBS:
        logs: deque = JOBS[job_id].get("logs")
        if logs is not None:
            logs.append(entry)

# ── Schemas ──────────────────────────────────────────────────────────────────
class SwapItem(BaseModel):
    target_indices: str  # e.g., "0" or "0,2"
    source_image: str    # URL

class SwapRequest(BaseModel):
    video_name: str
    webhook_url: Optional[str] = None
    swap: List[SwapItem]

class DetectFacesRequest(BaseModel):
    video_name: str

# ── Background job ───────────────────────────────────────────────────────────
def run_swap_job(job_id: str, req: SwapRequest):
    safe_update(job_id, status="running", progress=0, eta_seconds=None)
    try:
        uid = job_id
        video_path = f"videos/{req.video_name}.mp4"
        if not os.path.exists(video_path):
            raise RuntimeError(f"Video {req.video_name}.mp4 not found.")

        # Prepare dirs
        os.makedirs("source_images", exist_ok=True)
        os.makedirs("detected_faces", exist_ok=True)
        os.makedirs("output_videos", exist_ok=True)
        detected_dir = f"detected_faces/{req.video_name}"
        os.makedirs(detected_dir, exist_ok=True)

        output_raw = f"output_videos/{uid}_raw.mp4"
        final_output = f"output_videos/{uid}_final.mp4"

        # Source face
        swap_item = req.swap[0]
        source_img_path = f"source_images/{req.video_name}.jpg"
        job_log(job_id, f"Downloading source image: {swap_item.source_image}")
        download_image_from_url(swap_item.source_image, source_img_path)

        job_log(job_id, "Analyzing source face...")
        source_img = cv2.imread(source_img_path)
        source_faces = face_analyzer.get(source_img)
        if not source_faces:
            raise RuntimeError("No face found in source image.")
        source_face = source_faces[0]

        # First pass: detect unique faces
        job_log(job_id, "Scanning video for unique faces...")
        cap = cv2.VideoCapture(video_path)
        unique_faces = []
        threshold = 0.8
        N_FRAMES_SCAN = 300
        face_counter = 0

        for scan_idx in range(N_FRAMES_SCAN):
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
                    job_log(job_id, f"New face #{face_counter} detected & uploaded to S3.")
                    face_counter += 1

            if scan_idx and scan_idx % 100 == 0:
                job_log(job_id, f"Scanned {scan_idx}/{N_FRAMES_SCAN} frames for faces...")

        cap.release()
        if not unique_faces:
            raise RuntimeError("No unique faces detected in video.")

        target_ids = [int(i.strip()) for i in swap_item.target_indices.split(",") if i.strip().isdigit()]
        job_log(job_id, f"Target face indices: {target_ids}")

        # Second pass: swap
        job_log(job_id, "Starting face swap pass...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
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
                elapsed = max(1e-6, time.time() - start_time)
                progress = int((idx / total_frames) * 100)
                eta = (total_frames - idx) / (idx / elapsed) if idx else None
                safe_update(job_id, progress=progress, eta_seconds=round(eta, 1) if eta else None)
                job_log(job_id, f"Progress {progress}% ({idx}/{total_frames}) ETA≈{round(eta or 0,1)}s")

        cap.release()
        out.release()

        # Merge audio
        job_log(job_id, "Merging video with original audio via ffmpeg...")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", output_raw,
                "-i", video_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "libx264",
                "-c:a", "copy",
                "-movflags", "+faststart",
                final_output,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        # Upload final
        job_log(job_id, "Uploading final video to S3...")
        s3_video_path = f"output_videos/{uid}_final.mp4"
        video_url = upload_file_to_s3(final_output, s3_video_path, "video/mp4")
        job_log(job_id, f"Upload complete: {video_url}")

        # Optional webhook
        if req.webhook_url:
            try:
                job_log(job_id, f"Calling webhook: {req.webhook_url}")
                requests.post(req.webhook_url, json={"video_url": video_url, "job_id": job_id}, timeout=10)
            except Exception as e:
                job_log(job_id, f"Webhook failed: {e}")

        safe_update(
            job_id,
            status="done",
            progress=100,
            eta_seconds=0,
            result={"video_url": video_url, "swapped_frames": swapped_count},
        )
        job_log(job_id, "Job completed successfully.")

    except Exception as e:
        safe_update(job_id, status="error", error=str(e))
        job_log(job_id, f"Job error: {e}")

# ── Routes ───────────────────────────────────────────────────────────────────
@app.post("/swap")
async def swap_faces(req: SwapRequest, background: BackgroundTasks):
    if not req.swap:
        raise HTTPException(status_code=400, detail="swap[] cannot be empty")
    if not req.video_name:
        raise HTTPException(status_code=400, detail="video_name is required")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "result": None,
        "error": None,
        "eta_seconds": None,
        "request": json.loads(req.model_dump_json()),
        "logs": deque(maxlen=300),  # keep last 300 lines
    }
    job_log(job_id, f"Queued job for video '{req.video_name}'")
    background.add_task(run_swap_job, job_id, req)
    return {"message": "job queued", "job_id": job_id}

@app.get("/jobs/{job_id}")
async def get_job(job_id: str, include_logs: bool = Query(False)):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    payload = {
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress"),
        "eta_seconds": job.get("eta_seconds"),
        "result": job.get("result"),
        "error": job.get("error"),
    }
    if include_logs:
        logs: deque = job.get("logs") or deque()
        payload["logs"] = list(logs)
    return payload

@app.post("/detect-faces-in-video")
async def detect_faces_in_video(req: DetectFacesRequest):
    video_path = f"videos/{req.video_name}.mp4"
    detected_dir = f"detected_faces/{req.video_name}"
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video {req.video_name}.mp4 not found.")

    face_urls: List[str] = []
    if not os.path.exists(detected_dir) or not os.listdir(detected_dir):
        os.makedirs(detected_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        unique_faces = []
        threshold = 0.8
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
            raise HTTPException(status_code=400, detail="No unique faces detected.")
    else:
        for fname in sorted(os.listdir(detected_dir)):
            if fname.endswith(".jpg"):
                face_urls.append(
                    f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/detected_faces/{req.video_name}/{fname}"
                )
    return {"message": f"{len(face_urls)} face(s) detected", "face_images": face_urls}

# ─────────────────────────────────────────────────────────────────────────────
# Webhook test utilities
# ─────────────────────────────────────────────────────────────────────────────

class WebhookTestRequest(BaseModel):
    webhook_url: str
    job_id: Optional[str] = None         # if provided and job is done, we'll use its video_url
    video_url: Optional[str] = None      # override the video_url to send
    extra: Optional[Dict[str, Any]] = None  # any extra fields to merge into payload

@app.post("/test-webhook")
async def test_webhook(req: WebhookTestRequest):
    """
    Manually trigger a webhook for testing.

    Priority for video_url:
      1) req.video_url (if given)
      2) JOBS[req.job_id].result.video_url (if job_id provided and job done)
      3) fallback system example video
    """
    # Choose a job_id to send in payload (use provided or generate)
    payload_job_id = req.job_id or str(uuid.uuid4())

    # Resolve video_url
    video_url = req.video_url
    if not video_url and req.job_id and req.job_id in JOBS:
        job = JOBS[req.job_id]
        result = job.get("result") or {}
        video_url = result.get("video_url")

    if not video_url:
        # Fallback to a known public sample (adjust if you prefer)
        video_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/system_videos/EXAMPLE_OUT_Female.mp4"

    # Compose payload (same shape your real webhook uses)
    payload: Dict[str, Any] = {"video_url": video_url, "job_id": payload_job_id}
    if req.extra:
        payload.update(req.extra)

    try:
        r = requests.post(req.webhook_url, json=payload, timeout=10)
        # Try to present response meaningfully
        content_type = r.headers.get("content-type", "")
        try:
            body = r.json() if "application/json" in content_type else r.text[:1000]
        except Exception:
            body = r.text[:1000]
        return {
            "message": "Webhook sent",
            "target": req.webhook_url,
            "status_code": r.status_code,
            "payload": payload,
            "response": body,
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to call webhook: {e}")

@app.post("/jobs/{job_id}/replay-webhook")
async def replay_webhook(job_id: str, webhook_url: str = Query(..., description="Destination webhook URL")):
    """
    Re-send a webhook for a completed job using the stored result.video_url.
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.get("status") != "done":
        raise HTTPException(status_code=400, detail="job not completed yet")
    result = job.get("result") or {}
    video_url = result.get("video_url")
    if not video_url:
        raise HTTPException(status_code=400, detail="job has no video_url to send")

    payload = {"video_url": video_url, "job_id": job_id}
    try:
        r = requests.post(webhook_url, json=payload, timeout=10)
        content_type = r.headers.get("content-type", "")
        try:
            body = r.json() if "application/json" in content_type else r.text[:1000]
        except Exception:
            body = r.text[:1000]
        return {
            "message": "Webhook re-sent",
            "target": webhook_url,
            "status_code": r.status_code,
            "payload": payload,
            "response": body,
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to call webhook: {e}")

# (optional) health probe if you want it
@app.get("/health")
def health():
    return {"ok": True, "jobs": len(JOBS)}



# Run:
#   uvicorn main:app --host 0.0.0.0 --port 3000
# For quick parallelism on CPU: add --workers 2 (each worker has its own JOBS)
# For real multi-worker durability & logs: use Redis + Celery/RQ.
