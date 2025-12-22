import argparse
import os
import sys
import json
import subprocess
from pathlib import Path
from tqdm import tqdm
import math
import tempfile
import csv
import numpy as np

# video/audio libs
import cv2

# try to import whisper (preferred)
try:
    import whisper
except Exception:
    whisper = None

# pandas optional for nicer CSV
try:
    import pandas as pd
except Exception:
    pd = None

BASE = Path(__file__).resolve().parent

# Candidate modules (your uploaded files)
CAND_FACE_MODULES = [
    BASE / "realtimedetection.py",     # your webcam model
    BASE / "emotion_detector.py",
    BASE / "face_emotion.py",
    BASE / "emotiondetector.py",
]

TEXT_ANALYZER_PATH = BASE / "text_pipeline" / "analyzer.py"

# Helper: import a module by path
def import_module_from_path(path: Path):
    if not path.exists():
        return None
    name = f"user_module_{path.stem}"
    if name in sys.modules:
        return sys.modules[name]
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"[warn] executing {path.name} failed: {e}")
        return None
    sys.modules[name] = mod
    return mod

# Try to find a callable in a module by many possible names
def find_callable_in_module(mod):
    if not mod:
        return None
    candidates = [
        "predict_frame", "predict", "analyze_frame", "analyze_face",
        "detect_emotion", "detect_faces", "run", "main", "process_frame",
    ]
    for name in candidates:
        if hasattr(mod, name):
            attr = getattr(mod, name)
            if callable(attr):
                return attr
    return None

# Default lightweight face pipeline if your module not present:
def builtin_face_emotion(frame_bgr):
    # returns list of detections like [{'bbox':(x,y,w,h), 'emotion':'neutral', 'score':0.7}, ...]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, 1.1, 4)
    outs = []
    for (x,y,w,h) in faces:
        outs.append({"bbox": [int(x),int(y),int(w),int(h)], "emotion": "unknown", "score": None})
    return outs

# Try to load text analyzer
def get_text_analyzer():
    mod = import_module_from_path(TEXT_ANALYZER_PATH)
    if mod and hasattr(mod, "analyze_text") and callable(getattr(mod, "analyze_text")):
        return getattr(mod, "analyze_text")
    # fallback: try to use textblob if installed
    try:
        from textblob import TextBlob
        def analyze_text_tb(text):
            tb = TextBlob(text)
            return {"polarity": tb.sentiment.polarity, "subjectivity": tb.sentiment.subjectivity}
        return analyze_text_tb
    except Exception:
        return None

# Transcribe audio with whisper (if available), returning list of segments:
def transcribe_with_whisper(video_path, model_name="small"):
    if whisper is None:
        raise RuntimeError("whisper not installed. Install with `pip install -U openai-whisper`")
    model = whisper.load_model(model_name)
    # transcribe returns segments with start/end/text
    print("[info] transcribing with whisper model:", model_name)
    res = model.transcribe(str(video_path), verbose=False)
    segments = res.get("segments", [])
    # normalize segments to a list of {start,end,text}
    output = []
    for s in segments:
        output.append({"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()})
    return output

# If whisper missing, fallback to extracting full audio and trying other means (not implemented)
def extract_audio_ffmpeg(video_path, out_audio_path):
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(out_audio_path)
    ]
    subprocess.run(cmd, check=True)

def sample_frames_from_video(video_path, sample_rate_seconds=0.8):
    """
    Yields (timestamp_seconds, frame_bgr) sampled approx every sample_rate_seconds
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps else 0
    step_frames = max(1, int(round(fps * sample_rate_seconds)))
    frames = []
    frame_idx = 0
    pbar_total = int(math.ceil(total_frames / step_frames))+1
    pbar = tqdm(total=pbar_total, desc="Sampling frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step_frames == 0:
            ts = frame_idx / fps
            frames.append((ts, frame.copy()))
            pbar.update(1)
        frame_idx += 1
    pbar.close()
    cap.release()
    return frames

def align_faces_to_text(segments, face_records, window_sec=0.8):
    """
    segments: list of {"start","end","text", ...}
    face_records: list of {"timestamp":t, "faces":[...]}
    For each segment, gather face records where abs(face_ts - segment_mid) <= window_sec
    """
    out = []
    for seg in segments:
        mid = (seg["start"] + seg["end"]) / 2.0
        nearby = []
        for rec in face_records:
            if abs(rec["timestamp"] - mid) <= window_sec:
                nearby.extend(rec["faces"])
        out.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "text_analysis": seg.get("analysis"),
            "faces": nearby
        })
    return out

def main():
    ap = argparse.ArgumentParser(description="Video -> combined text+facial emotion analysis")
    ap.add_argument("video", help="path to video file")
    ap.add_argument("--out", "-o", default="video_emotion_output.json")
    ap.add_argument("--csv", action="store_true", help="also write CSV")
    ap.add_argument("--sample-rate", type=float, default=0.8, help="seconds between sampled frames (default 0.8s)")
    ap.add_argument("--whisper-model", default="small", help="whisper model name if installed")
    ap.add_argument("--face-window", type=float, default=0.8, help="seconds window to align faces to transcript mid")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print("Video not found:", video_path)
        sys.exit(1)

    # 1) Transcribe
    segments = []
    if whisper is not None:
        try:
            segments = transcribe_with_whisper(video_path, model_name=args.whisper_model)
        except Exception as e:
            print("[warn] whisper transcription failed:", e)
    else:
        print("[warn] whisper not installed. Try: pip install -U openai-whisper")
        # fallback: no transcription -> single empty segment spanning video
        # try to get duration from cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps else 0
        cap.release()
        segments = [{"start": 0.0, "end": duration, "text": ""}]
        print(f"[info] created empty single segment of duration {duration}s")

    print(f"[info] got {len(segments)} transcript segments")

    # 2) load text analyzer
    analyze_text = get_text_analyzer()
    if analyze_text is None:
        print("[warn] No text analyzer found. Install textblob or provide text_pipeline/analyzer.py with analyze_text(text).")

    # 3) annotate segments with text analysis
    for seg in segments:
        if analyze_text and seg.get("text"):
            try:
                seg["analysis"] = analyze_text(seg["text"])
            except Exception as e:
                seg["analysis"] = {"error": str(e)}

    # 4) sample frames and run face detector
    frames = sample_frames_from_video(video_path, sample_rate_seconds=args.sample_rate)

    # prepare face detection function
    face_detector_fn = None
    for cand in CAND_FACE_MODULES:
        mod = import_module_from_path(cand)
        fn = find_callable_in_module(mod)
        if fn:
            print(f"[info] using face/emotion function {fn.__name__} from {cand.name}")
            face_detector_fn = fn
            break
    if face_detector_fn is None:
        print("[info] no face/emotion module found; using lightweight Haar cascade fallback.")
        face_detector_fn = builtin_face_emotion

    face_records = []
    for ts, frame in tqdm(frames, desc="Running face detector"):
        # run detector; try to handle both (frame) and (image_path) call signatures
        faces = []
        try:
            # prefer calling with frame (BGR numpy array)
            out = face_detector_fn(frame)
            # normalize output
            if out is None:
                faces = []
            elif isinstance(out, list):
                faces = out
            elif isinstance(out, dict):
                faces = [out]
            else:
                faces = out
        except TypeError:
            # maybe it expects a path: write temp image
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
                cv2.imwrite(tmp.name, frame)
                try:
                    out = face_detector_fn(tmp.name)
                    if out is None:
                        faces = []
                    elif isinstance(out, list):
                        faces = out
                    elif isinstance(out, dict):
                        faces = [out]
                    else:
                        faces = out
                except Exception as e:
                    print("[warn] face detector call failed on path:", e)
                    faces = []
        except Exception as e:
            print("[warn] face detector exception:", e)
            faces = []

        face_records.append({"timestamp": ts, "faces": faces})

    # 5) align faces to text segments
    combined = align_faces_to_text(segments, face_records, window_sec=args.face_window)

    # 6) write outputs
    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf8") as f:
        json.dump({"video": str(video_path), "combined": combined}, f, indent=2, ensure_ascii=False)
    print("[info] saved JSON ->", out_path)

    if args.csv:
        rows = []
        for seg in combined:
            row = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "text_analysis": json.dumps(seg.get("text_analysis"), ensure_ascii=False),
                "num_faces": len(seg.get("faces", [])),
                "faces": json.dumps(seg.get("faces", []), ensure_ascii=False)
            }
            rows.append(row)
        csv_path = out_path.with_suffix(".csv")
        if pd is not None:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
        else:
            with open(csv_path, "w", newline="", encoding="utf8") as cf:
                writer = csv.DictWriter(cf, fieldnames=list(rows[0].keys()) if rows else ["start","end","text"])
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
        print("[info] saved CSV ->", csv_path)

if __name__ == "__main__":
    main()
