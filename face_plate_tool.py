# -*- coding: utf-8 -*-
import os, sys, threading, time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable

import cv2
import numpy as np
from PIL import Image, ImageTk

import torch
from ultralytics import YOLO


# ===================== UTILITIES =====================

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

@dataclass
class AppConfig:
    imgsz: int = 960
    conf: float = 0.35      # detec. conf threshold (new tracks / output)
    keep_conf: float = 0.15 # low-threshold to keep weak dets (for tracking)
    iou_nms: float = 0.45
    pad_ratio: float = 0.18
    iou_match_th: float = 0.30
    max_missed: int = 8
    smooth_alpha: float = 0.6
    blur_ksize: int = 35
    blur_sigma: int = 0


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def safe_load(weights: str) -> YOLO:
    m = YOLO(weights)
    # fuse() bəzi hallarda layout warning verə bilər, amma ümumilikdə sürəti artırır
    try:
        m.fuse()
    except Exception:
        pass
    return m


def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w - 1, int(x2)))
    y2 = max(0, min(h - 1, int(y2)))
    return x1, y1, x2, y2


def pad_box(x1, y1, x2, y2, w, h, ratio=0.15):
    bw, bh = (x2 - x1), (y2 - y1)
    px, py = int(bw * ratio), int(bh * ratio)
    return clamp_box(x1 - px, y1 - py, x2 + px, y2 + py, w, h)


def draw_label(img, x, y, text):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 6, y), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)


def predict_boxes(model: YOLO, img_bgr, device: str, w: int, h: int, label: str,
                  imgsz: int, conf: float, iou: float) -> List[Tuple[str, Tuple[int,int,int,int], float]]:
    # Ultralytics BGR-ni özü convert edir, numpy array daxil etmək olur
    preds = model.predict(
        img_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
        device=device,
        half=(device.startswith("cuda"))
    )
    out = []
    for r in preds:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        for b in r.boxes:
            score = float(b.conf[0]) if (b.conf is not None and len(b.conf) > 0) else 1.0
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
            if (x2 - x1) < 6 or (y2 - y1) < 6:
                continue
            out.append((label, (x1, y1, x2, y2), score))
    return out


# ===================== SIMPLE TRACKER =====================

class Track:
    _next_id = 1
    def __init__(self, label: str, box: Tuple[int,int,int,int], conf: float, alpha=0.6):
        self.id = Track._next_id; Track._next_id += 1
        self.label = label
        self.box = box
        self.conf = conf
        self.missed = 0
        self.alpha = alpha

    def update(self, new_box, conf):
        px1, py1, px2, py2 = self.box
        nx1, ny1, nx2, ny2 = new_box
        a = self.alpha
        x1 = int(a * nx1 + (1 - a) * px1)
        y1 = int(a * ny1 + (1 - a) * py1)
        x2 = int(a * nx2 + (1 - a) * px2)
        y2 = int(a * ny2 + (1 - a) * py2)
        self.box = (x1, y1, x2, y2)
        self.conf = conf
        self.missed = 0


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    aarea = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    barea = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = aarea + barea - inter
    return inter / union if union > 0 else 0.0


def match_tracks(tracks: List[Track],
                 detections: List[Tuple[str, Tuple[int,int,int,int], float]],
                 iou_th: float,
                 keep_conf: float,
                 det_conf: float,
                 alpha: float):
    used = set()
    # update existing
    for t in tracks:
        best_j, best_iou = -1, 0.0
        for j, (lbl, box, conf) in enumerate(detections):
            if j in used or lbl != t.label:
                continue
            i = iou_xyxy(t.box, box)
            if i > best_iou:
                best_iou, best_j = i, j
        if best_j >= 0 and best_iou >= iou_th:
            lbl, box, conf = detections[best_j]
            t.alpha = alpha
            t.update(box, conf)
            used.add(best_j)
        else:
            t.missed += 1

    # create new
    for j, (lbl, box, conf) in enumerate(detections):
        if j in used:
            continue
        if conf >= det_conf:
            tracks.append(Track(lbl, box, conf, alpha=alpha))

    return tracks


def prune_tracks(tracks: List[Track], max_missed: int) -> List[Track]:
    return [t for t in tracks if t.missed <= max_missed]


def blur_regions(frame, boxes: List[Tuple[int,int,int,int]], ksize=35, sigma=0):
    k = (ksize if isinstance(ksize, tuple) else (ksize, ksize))
    for (x1, y1, x2, y2) in boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, k, sigma)


# ===================== TK APP =====================

class FacePlateApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face & Plate — Box / Blur Tool (Tkinter)")
        self.geometry("1080x720")

        self.cfg = AppConfig()
        self.device = pick_device()

        # models
        self.face_weights = tk.StringVar(value="")
        self.plate_weights = tk.StringVar(value="")
        self.model_face: Optional[YOLO] = None
        self.model_plate: Optional[YOLO] = None

        # options
        self.mode_var = tk.StringVar(value="image")      # image or video
        self.op_var = tk.StringVar(value="bbox")         # bbox or blur
        self.use_face = tk.BooleanVar(value=True)
        self.use_plate = tk.BooleanVar(value=True)
        self.imgsz_var = tk.IntVar(value=self.cfg.imgsz)
        self.conf_var = tk.DoubleVar(value=self.cfg.conf)
        self.iou_var = tk.DoubleVar(value=self.cfg.iou_nms)
        self.pad_var = tk.DoubleVar(value=self.cfg.pad_ratio)
        self.ksize_var = tk.IntVar(value=self.cfg.blur_ksize)
        self.out_path = tk.StringVar(value="")

        # state
        self.input_path: Optional[str] = None
        self.preview_imgtk = None  # keep ref
        self.running_thread: Optional[threading.Thread] = None
        self.cancel_flag = False

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.Frame(self); top.pack(fill="x", padx=10, pady=8)

        ttk.Label(top, text=f"Device: {self.device}").grid(row=0, column=0, sticky="w")

        # weights row
        row1 = ttk.Frame(self); row1.pack(fill="x", padx=10, pady=4)
        ttk.Label(row1, text="Face weights:").grid(row=0, column=0, sticky="w")
        ttk.Entry(row1, textvariable=self.face_weights, width=70).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(row1, text="Seç", command=self.choose_face_w).grid(row=0, column=2, padx=2)

        ttk.Label(row1, text="Plate weights:").grid(row=1, column=0, sticky="w")
        ttk.Entry(row1, textvariable=self.plate_weights, width=70).grid(row=1, column=1, sticky="we", padx=4)
        ttk.Button(row1, text="Seç", command=self.choose_plate_w).grid(row=1, column=2, padx=2)

        # controls
        row2 = ttk.Frame(self); row2.pack(fill="x", padx=10, pady=6)
        ttk.Label(row2, text="Mode:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(row2, text="Image", variable=self.mode_var, value="image").grid(row=0, column=1)
        ttk.Radiobutton(row2, text="Video", variable=self.mode_var, value="video").grid(row=0, column=2)

        ttk.Label(row2, text="Operation:").grid(row=0, column=3, sticky="w", padx=(18,0))
        ttk.Radiobutton(row2, text="Bounding Box", variable=self.op_var, value="bbox").grid(row=0, column=4)
        ttk.Radiobutton(row2, text="Blur", variable=self.op_var, value="blur").grid(row=0, column=5)

        ttk.Checkbutton(row2, text="Face", variable=self.use_face).grid(row=0, column=6, padx=(18,0))
        ttk.Checkbutton(row2, text="Plate", variable=self.use_plate).grid(row=0, column=7)

        # params
        row3 = ttk.Frame(self); row3.pack(fill="x", padx=10, pady=4)
        ttk.Label(row3, text="imgsz").grid(row=0, column=0); ttk.Entry(row3, textvariable=self.imgsz_var, width=6).grid(row=0, column=1)
        ttk.Label(row3, text="conf").grid(row=0, column=2); ttk.Entry(row3, textvariable=self.conf_var, width=6).grid(row=0, column=3)
        ttk.Label(row3, text="iou").grid(row=0, column=4); ttk.Entry(row3, textvariable=self.iou_var, width=6).grid(row=0, column=5)
        ttk.Label(row3, text="pad").grid(row=0, column=6); ttk.Entry(row3, textvariable=self.pad_var, width=6).grid(row=0, column=7)
        ttk.Label(row3, text="blur k").grid(row=0, column=8); ttk.Entry(row3, textvariable=self.ksize_var, width=6).grid(row=0, column=9)

        # file & run
        row4 = ttk.Frame(self); row4.pack(fill="x", padx=10, pady=8)
        ttk.Button(row4, text="Fayl seç (şəkil/video)", command=self.choose_input).grid(row=0, column=0)
        ttk.Button(row4, text="Model(ləri) yüklə", command=self.load_models).grid(row=0, column=1, padx=6)
        ttk.Button(row4, text="İcra et", command=self.run_clicked).grid(row=0, column=2)
        ttk.Button(row4, text="Dayandır", command=self.stop_clicked).grid(row=0, column=3, padx=4)

        ttk.Label(row4, text="Çıxış faylı:").grid(row=0, column=4, padx=(16,4))
        ttk.Entry(row4, textvariable=self.out_path, width=40).grid(row=0, column=5, sticky="we")
        ttk.Button(row4, text="Yol seç", command=self.choose_output).grid(row=0, column=6, padx=4)

        # progress
        self.prog = ttk.Progressbar(self, orient="horizontal", mode="determinate")
        self.prog.pack(fill="x", padx=10, pady=6)

        # preview canvas
        self.preview = ttk.Label(self)
        self.preview.pack(fill="both", expand=True, padx=10, pady=6)
        self.preview.configure(anchor="center", text="Şəkil önbaxış burada görünəcək\n(Video üçün ilk kadr göstərilir)")

        # status
        self.status_var = tk.StringVar(value="Hazır.")
        ttk.Label(self, textvariable=self.status_var).pack(fill="x", padx=10, pady=(0,10))

    # ---------- callbacks ----------
    def choose_face_w(self):
        p = filedialog.askopenfilename(title="Face .pt seç", filetypes=[("PyTorch weights", "*.pt")])
        if p: self.face_weights.set(p)

    def choose_plate_w(self):
        p = filedialog.askopenfilename(title="Plate .pt seç", filetypes=[("PyTorch weights", "*.pt")])
        if p: self.plate_weights.set(p)

    def choose_input(self):
        p = filedialog.askopenfilename(title="Şəkil və ya Video seç",
                                       filetypes=[("Media", "*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tif;*.tiff;*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v")])
        if p:
            self.input_path = p
            self.status_var.set(f"Giriş: {p}")
            if is_image(p):
                self.show_image_preview(p)
            elif is_video(p):
                self.show_video_first_frame(p)

    def choose_output(self):
        if self.mode_var.get() == "video":
            p = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi")])
        else:
            p = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if p:
            self.out_path.set(p)

    def load_models(self):
        try:
            used = []
            if self.use_face.get():
                if not self.face_weights.get():
                    raise ValueError("Face weights yolu boşdur.")
                self.model_face = safe_load(self.face_weights.get())
                used.append("Face")
            else:
                self.model_face = None
            if self.use_plate.get():
                if not self.plate_weights.get():
                    raise ValueError("Plate weights yolu boşdur.")
                self.model_plate = safe_load(self.plate_weights.get())
                used.append("Plate")
            else:
                self.model_plate = None
            if not used:
                raise ValueError("Heç bir model seçilməyib (Face / Plate).")
            self.status_var.set(f"Yükləndi: {', '.join(used)} (device: {self.device})")
            messagebox.showinfo("OK", "Model(lər) yükləndi.")
        except Exception as e:
            messagebox.showerror("Xəta", str(e))

    def run_clicked(self):
        if self.running_thread and self.running_thread.is_alive():
            messagebox.showwarning("Gözlə", "İcra davam edir.")
            return
        if not self.input_path:
            messagebox.showwarning("Fayl", "Əvvəlcə şəkil/video seçin.")
            return
        if (self.use_face.get() and self.model_face is None) or (self.use_plate.get() and self.model_plate is None):
            messagebox.showwarning("Model", "Əvvəlcə model(ləri) yükləyin.")
            return

        self.cancel_flag = False
        if is_image(self.input_path):
            self.running_thread = threading.Thread(target=self.process_image, daemon=True)
        else:
            self.running_thread = threading.Thread(target=self.process_video, daemon=True)
        self.running_thread.start()

    def stop_clicked(self):
        self.cancel_flag = True
        self.status_var.set("Dayandırma istənildi...")

    # ---------- preview helpers ----------
    def show_image_preview(self, path):
        try:
            img = Image.open(path).convert("RGB")
            self._set_preview(img)
        except Exception as e:
            self.preview.configure(text=f"Şəkli açmaq olmadı: {e}")

    def show_video_first_frame(self, path):
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            self.preview.configure(text="Videonun ilk kadrını oxumaq olmadı.")
            return
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._set_preview(img)

    def _set_preview(self, img: Image.Image):
        # resize to fit label
        w0 = self.preview.winfo_width() or 960
        h0 = self.preview.winfo_height() or 540
        img = img.copy()
        img.thumbnail((w0, h0))
        self.preview_imgtk = ImageTk.PhotoImage(img)
        self.preview.configure(image=self.preview_imgtk, text="")

    # ---------- core processing ----------
    def process_image(self):
        try:
            img = cv2.imread(self.input_path)
            if img is None:
                raise ValueError("Şəkli oxumaq olmadı.")
            H, W = img.shape[:2]

            imgsz = int(self.imgsz_var.get())
            conf = float(self.conf_var.get())
            iou = float(self.iou_var.get())
            padr = float(self.pad_var.get())
            ksize = int(self.ksize_var.get())

            detections: List[Tuple[str, Tuple[int,int,int,int], float]] = []

            if self.use_face.get() and self.model_face is not None:
                detections += predict_boxes(self.model_face, img, self.device, W, H, "face", imgsz, conf, iou)

            if self.use_plate.get() and self.model_plate is not None:
                detections += predict_boxes(self.model_plate, img, self.device, W, H, "plate", imgsz, conf, iou)

            out_img = img.copy()
            if self.op_var.get() == "blur":
                boxes = [pad_box(*b, W, H, padr) for _, b, _ in detections]
                blur_regions(out_img, boxes, ksize=ksize, sigma=0)
            else:
                for lbl, (x1, y1, x2, y2), sc in detections:
                    x1, y1, x2, y2 = pad_box(x1, y1, x2, y2, W, H, padr)
                    cv2.rectangle(out_img, (x1, y1), (x2, y2), (0,255,255), 2)
                    draw_label(out_img, x1, y1, f"{lbl} {sc:.2f}")

            # save or preview
            if self.out_path.get():
                ok = cv2.imwrite(self.out_path.get(), out_img)
                if not ok:
                    raise ValueError("Çıxış şəkli saxlanmadı.")
                self.status_var.set(f"Yazıldı: {self.out_path.get()}")
            else:
                # preview
                self.status_var.set("Tamam — önbaxış göstərilir.")
                pil = Image.fromarray(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
                self._set_preview(pil)

        except Exception as e:
            messagebox.showerror("Xəta", str(e))

    def process_video(self):
        try:
            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                raise ValueError("Videonu açmaq olmadı.")

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if not self.out_path.get():
                # default output eyni qovluqda `_out.mp4`
                base, ext = os.path.splitext(self.input_path)
                self.out_path.set(base + "_out.mp4")

            out_path = self.out_path.get()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
            if not writer.isOpened():
                # fallback
                out_path = base + "_out.avi"
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (W, H))
                if not writer.isOpened():
                    raise ValueError("VideoWriter açıla bilmədi (mp4/avi).")

            # params
            imgsz = int(self.imgsz_var.get())
            conf = float(self.conf_var.get())
            iou = float(self.iou_var.get())
            padr = float(self.pad_var.get())
            ksize = int(self.ksize_var.get())

            # tracking structures
            tracks: List[Track] = []
            iou_match_th = self.cfg.iou_match_th
            keep_conf = self.cfg.keep_conf
            det_conf = conf
            max_missed = self.cfg.max_missed
            alpha = self.cfg.smooth_alpha

            self.prog.configure(maximum=(total if total > 0 else 100))
            self.prog["value"] = 0

            frame_idx = 0
            last_preview_time = 0

            while True:
                if self.cancel_flag:
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                dets = []

                if self.use_face.get() and self.model_face is not None:
                    dets += predict_boxes(self.model_face, frame, self.device, W, H, "face", imgsz, keep_conf, iou)
                if self.use_plate.get() and self.model_plate is not None:
                    dets += predict_boxes(self.model_plate, frame, self.device, W, H, "plate", imgsz, keep_conf, iou)

                tracks = match_tracks(tracks, dets, iou_match_th, keep_conf, det_conf, alpha)
                tracks = prune_tracks(tracks, max_missed)

                if self.op_var.get() == "blur":
                    boxes = [pad_box(*t.box, W, H, padr) for t in tracks]
                    blur_regions(frame, boxes, ksize=ksize, sigma=0)
                else:
                    for t in tracks:
                        x1, y1, x2, y2 = pad_box(*t.box, W, H, padr)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
                        draw_label(frame, x1, y1, f"{t.label} id{t.id} {t.conf:.2f}")

                writer.write(frame)

                frame_idx += 1
                if total > 0:
                    self.prog["value"] = frame_idx

                # hər ~0.3s bir önbaxış yenilə
                now = time.time()
                if now - last_preview_time > 0.3:
                    last_preview_time = now
                    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    self._set_preview(pil)
                    self.update_idletasks()

            writer.release()
            cap.release()
            self.status_var.set("Tamamlandı." if not self.cancel_flag else "Dayandırıldı.")
            if self.cancel_flag:
                messagebox.showinfo("Dayandırıldı", f"Yarımçıq çıxış: {out_path}")
            else:
                messagebox.showinfo("OK", f"Yazıldı: {out_path}")

        except Exception as e:
            messagebox.showerror("Xəta", str(e))


# ===================== MAIN =====================

def main():
    app = FacePlateApp()
    # İstəyirsənsə default weight yollarını bura yaz:
    # app.face_weights.set(r"D:\MyPc\Desktop\Data\FacePlateBlur\face_best.pt")
    # app.plate_weights.set(r"D:\MyPc\Desktop\Data\FacePlateBlur\plate_best_xtreme.pt")
    app.mainloop()


if __name__ == "__main__":
    main()
