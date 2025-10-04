# -*- coding: utf-8 -*-
import os, sys, math
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
from typing import List, Tuple, Iterable, Optional
from tqdm import tqdm

# --------- İSTİFADƏÇİ AYARLARI ----------
FACE_WEIGHTS  = r"D:\MyPc\Desktop\Data\FacePlateBlur\face_best.pt"
PLATE_WEIGHTS = r"D:\MyPc\Desktop\Data\FacePlateBlur\plate_best_xtreme.pt"

VIDEO_DIR     = Path(r"D:\MyPc\Desktop\Data\FacePlateBlur\Sample Videos")
OUT_DIR       = Path(r"D:\MyPc\Desktop\Data\FacePlateBlur\Blurred Videos")

IMGSZ   = 960
IOU_NMS = 0.45

# Hysteresis thresholds
DET_CONF  = 0.35   # yeni track yaratmaq üçün hədd
KEEP_CONF = 0.15   # mövcud track-ı saxlamaq üçün minimal güvən

# Tracking/smoothing
MAX_MISSED    = 8          # obyekt görünməsə də bu qədər kadr blur davam etsin
IOU_MATCH_TH  = 0.3        # track ilə yeni deteksiyanı eşləşdirmə həddi
SMOOTH_ALPHA  = 0.6        # EMA: yeni=alpha*det + (1-alpha)*köhnə
PAD_RATIO     = 0.18       # blur qutusunu hər tərəfə % genişləndir
MIN_BOX_WH    = 6          # piksel: çox balaca qutuları at

SHOW_BOX = False
LABEL_FACE  = "face"
LABEL_PLATE = "plate"

BLUR_KSIZE = (35, 35)
BLUR_SIGMA = 0

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
# ---------------------------------------

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def safe_load(weights: str) -> YOLO:
    m = YOLO(weights)
    m.fuse()
    return m

# -------------------- Utility --------------------
def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aarea = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    barea = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = aarea + barea - inter
    return inter / union if union > 0 else 0.0

def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w-1, int(x1)))
    y1 = max(0, min(h-1, int(y1)))
    x2 = max(0, min(w-1, int(x2)))
    y2 = max(0, min(h-1, int(y2)))
    return x1, y1, x2, y2

def pad_box(x1, y1, x2, y2, w, h, ratio=0.15):
    bw, bh = (x2 - x1), (y2 - y1)
    px, py = int(bw * ratio), int(bh * ratio)
    return clamp_box(x1 - px, y1 - py, x2 + px, y2 + py, w, h)

def ema_box(prev, new, alpha=0.6):
    if prev is None:
        return new
    px1, py1, px2, py2 = prev
    nx1, ny1, nx2, ny2 = new
    x1 = int(alpha * nx1 + (1 - alpha) * px1)
    y1 = int(alpha * ny1 + (1 - alpha) * py1)
    x2 = int(alpha * nx2 + (1 - alpha) * px2)
    y2 = int(alpha * ny2 + (1 - alpha) * py2)
    return x1, y1, x2, y2

def draw_label(img, x, y, text):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 6, y), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

# -------------------- Simple Tracker --------------------
class Track:
    _next_id = 1
    def __init__(self, label: str, box: Tuple[int,int,int,int], conf: float):
        self.id = Track._next_id; Track._next_id += 1
        self.label = label
        self.box = box            # (x1,y1,x2,y2)
        self.conf = conf
        self.missed = 0

    def update(self, box, conf):
        self.box = ema_box(self.box, box, SMOOTH_ALPHA)
        self.conf = conf
        self.missed = 0

def match_tracks(tracks: List[Track], detections: List[Tuple[str,Tuple[int,int,int,int],float]]) -> None:
    """Greedy IOU matching by label."""
    used = set()
    for t in tracks:
        best_j, best_iou = -1, 0.0
        for j, (lbl, box, conf) in enumerate(detections):
            if j in used or lbl != t.label:
                continue
            i = iou_xyxy(t.box, box)
            if i > best_iou:
                best_iou, best_j = i, j
        if best_j >= 0 and best_iou >= IOU_MATCH_TH:
            lbl, box, conf = detections[best_j]
            t.update(box, conf)
            used.add(best_j)
        else:
            t.missed += 1
    # create new tracks for unmatched detections above DET_CONF
    for j, (lbl, box, conf) in enumerate(detections):
        if j in used:
            continue
        if conf >= DET_CONF:
            tracks.append(Track(lbl, box, conf))

def prune_tracks(tracks: List[Track]) -> List[Track]:
    return [t for t in tracks if t.missed <= MAX_MISSED]

# -------------------- Inference helpers --------------------
def predict_boxes(model: YOLO, img, device: str, w: int, h: int, label: str) -> List[Tuple[str,Tuple[int,int,int,int],float]]:
    preds = model.predict(img, imgsz=IMGSZ, conf=KEEP_CONF, iou=IOU_NMS, verbose=False, device=device, half=(device.startswith("cuda")))
    out: List[Tuple[str,Tuple[int,int,int,int],float]] = []
    for r in preds:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        for b in r.boxes:
            conf = float(b.conf[0]) if b.conf is not None else 1.0
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
            if (x2 - x1) < MIN_BOX_WH or (y2 - y1) < MIN_BOX_WH:
                continue
            out.append((label, (x1, y1, x2, y2), conf))
    return out

def blur_regions(frame, boxes: List[Tuple[int, int, int, int]]):
    for (x1, y1, x2, y2) in boxes:
        if x2 <= x1 or y2 <= y1: 
            continue
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, BLUR_KSIZE, BLUR_SIGMA)

# -------------------- Video loop --------------------
def iter_videos(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p

def make_out_path(in_path: Path, in_root: Path, out_root: Path) -> Path:
    rel = in_path.relative_to(in_root)
    out_path = out_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path

def process_single_video(in_path: Path, out_path: Path, face_model: YOLO, plate_model: YOLO, device: str):
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"[!] Açıla bilmədi: {in_path}")
        return

    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not out.isOpened():
        alt_out = out_path.with_suffix(".avi")
        print(f"[i] mp4 açılmadı, .avi sınaq: {alt_out.name}")
        out = cv2.VideoWriter(str(alt_out), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
        if not out.isOpened():
            print(f"[!] Çıxış yazıcı açıla bilmədi: {out_path}")
            cap.release()
            return

    # Track dəstləri: eyni tracker daxilində label-ə görə ayrılır
    tracks: List[Track] = []

    pbar = tqdm(total=(total_frames if total_frames > 0 else None), desc=f"İşlənir: {in_path.name}", unit="frm")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1) Detections (zəifləri də götürürük; saxlamanı tracker edəcək)
        dets_face  = predict_boxes(face_model,  frame, device, w, h, LABEL_FACE)
        dets_plate = predict_boxes(plate_model, frame, device, w, h, LABEL_PLATE)
        detections = dets_face + dets_plate

        # 2) Track update + yaradılma
        match_tracks(tracks, detections)
        tracks = prune_tracks(tracks)

        # 3) Çıxış qutuları: pad + blur
        draw_boxes = []
        for t in tracks:
            x1, y1, x2, y2 = pad_box(*t.box, w, h, PAD_RATIO)
            draw_boxes.append((x1, y1, x2, y2))
            if SHOW_BOX:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
                draw_label(frame, x1, y1, f"{t.label} id{t.id} m{t.missed}")

        blur_regions(frame, draw_boxes)

        out.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

def main():
    device = pick_device()
    print(f"[i] Device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    face_model  = safe_load(FACE_WEIGHTS)
    plate_model = safe_load(PLATE_WEIGHTS)

    videos = list(iter_videos(VIDEO_DIR))
    if not videos:
        print(f"[!] Video tapılmadı: {VIDEO_DIR}")
        return

    print(f"[i] {len(videos)} video tapıldı.")
    for v in videos:
        out_path = make_out_path(v, VIDEO_DIR, OUT_DIR)
        print(f"\n[i] Giriş: {v}\n[i] Çıxış: {out_path}")
        process_single_video(v, out_path, face_model, plate_model, device)

    print("\n[✓] Bütün videolar emal edildi.")

if __name__ == "__main__":
    main()
