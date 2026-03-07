import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").setLevel(logging.WARNING)

import gradio as gr
from ultralytics import YOLO
import numpy as np
import cv2

# Load model - replace with your fine-tuned weights if available
model = YOLO("yolov8n_07mar261756.pt")

# Your 2 classes with BGR colors
class_names = ['normal_package', 'damaged_package']
colors = [
    (0, 255, 0),   # Green for normal_package
    (255, 0, 0),   # Red for damaged_package
]

def detect(image, conf_threshold):
    if image is None:
        return None, "⏳ Waiting for webcam..."

    if isinstance(image, dict):
        image = image.get("composite", image.get("image", None))
    if image is None:
        return None, "⏳ Waiting for webcam..."

    results = model.predict(image, conf=conf_threshold, verbose=False)

    frame = image.copy()
    counts = {}

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = round(float(box.conf[0]), 2)
            class_id = int(box.cls[0])

            if 0 <= class_id < len(class_names):
                class_name = class_names[class_id]
                color = colors[class_id]
            else:
                continue  # Skip all non-package classes

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{class_name}: {confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            counts[class_name] = counts.get(class_name, 0) + 1

    # Build summary
    if counts:
        parts = []
        if "normal_package" in counts:
            parts.append(f"✅ Normal: {counts['normal_package']}")
        if "damaged_package" in counts:
            parts.append(f"🔴 Damaged: {counts['damaged_package']}")
        summary = " | ".join(parts)
    else:
        summary = "📭 No packages detected"

    return frame, summary


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
                label="📷 Webcam Input"
            )
            conf_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.25,
                step=0.05,
                label="Confidence Threshold"
            )
            gr.Markdown("⬆️ Lower = more detections, Higher = fewer but more certain")

        with gr.Column(scale=1):
            output_img = gr.Image(label="🎯 Detection Output")
            output_text = gr.Textbox(
                label="📊 Detection Summary",
                placeholder="Waiting for detections...",
            )

    webcam.stream(
        fn=detect,
        inputs=[webcam, conf_slider],
        outputs=[output_img, output_text],
        stream_every=1
    )

demo.launch(ssr_mode=False)
