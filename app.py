import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").setLevel(logging.WARNING)

import gradio as gr
from ultralytics import YOLO
import numpy as np
import cv2

# Load model
model = YOLO("best.pt")

class_names = ['normal_package', 'damaged_package']
colors = [
    (0, 255, 0),    # Green for normal_package  (RGB)
    (255, 0, 0),    # Red   for damaged_package (RGB)
]

def detect(image, conf_threshold):
    # ── Guard: no frame yet ───────────────────────────────────────
    if image is None:
        return None, "⏳ Waiting for webcam..."

    # ── Handle dict input from Gradio editor component ────────────
    if isinstance(image, dict):
        image = image.get("composite", image.get("image", None))
    if image is None:
        return None, "⏳ Waiting for webcam..."

    # ── Core variables (must be initialised before the loop) ──────
    frame      = image.copy()
    h, w       = frame.shape[:2]
    frame_area = h * w
    counts     = {}

    # ── Run inference ─────────────────────────────────────────────
    results = model.predict(image, conf=conf_threshold, verbose=False)

    for r in results:
        for box in r.boxes:
            # Extract values BEFORE using them
            x1, y1, x2, y2 = map(int, box.xyxy[0])   # int for cv2
            class_id        = int(box.cls[0])
            confidence      = float(box.conf[0])

            # ── Reject boxes that cover >60 % of frame ────────────
            box_area = (x2 - x1) * (y2 - y1)
            if box_area / frame_area > 0.6:
                continue          # too large → almost certainly not a parcel

            # ── Skip unknown classes ──────────────────────────────
            if not (0 <= class_id < len(class_names)):
                continue

            class_name = class_names[class_id]
            color      = colors[class_id]

            # ── Draw bounding box ─────────────────────────────────
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ── Draw label background + text ─────────────────────
            text = f"{class_name}: {confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - th - 10),
                (x1 + tw, y1),
                color, -1
            )
            cv2.putText(
                frame, text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

            counts[class_name] = counts.get(class_name, 0) + 1

    # ── Build summary text ────────────────────────────────────────
    if counts:
        parts = []
        if "normal_package"  in counts:
            parts.append(f"✅ Normal count : {counts['normal_package']}")
        if "damaged_package" in counts:
            parts.append(f"🔴 Damaged count: {counts['damaged_package']}")
        summary = " | ".join(parts)
    else:
        summary = "📭 No packages detected"

    return frame, summary


# ── Gradio UI ─────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📦 Package Damage Detection
    **Live webcam detection** — click Record to start, then detections update every second.
    > 🟢 Green = Normal Package &nbsp;&nbsp; 🔴 Red = Damaged Package
    """)

    with gr.Row():
        with gr.Column(scale=1):
            webcam = gr.Image(
                sources=["webcam"],
                type="numpy",
                streaming=True,
                label="📷 Webcam Input",
            )
            conf_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.75,
                step=0.05,
                label="Confidence Threshold",
            )
            gr.Markdown("⬆️ Lower = more detections · Higher = fewer but more certain")

        with gr.Column(scale=1):
            output_img  = gr.Image(label="🎯 Detection Output")
            output_text = gr.Textbox(
                label="📊 Detection Summary",
                placeholder="Waiting for detections...",
            )

    webcam.stream(
        fn=detect,
        inputs=[webcam, conf_slider],
        outputs=[output_img, output_text],
        stream_every=1,
    )

demo.launch(ssr_mode=False)
