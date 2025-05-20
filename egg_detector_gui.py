import gradio as gr
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from datetime import datetime
import os
from collections import deque
import time
import tensorflow as tf

# Store (timestamp, image) tuples
frame_history = deque()

model = load_model(r"D:\code\py\app.py\egg_detector_model.h5")
log_file = "results.txt"

# ------------------ Modular Helper Functions ------------------

def detect_cleanliness_and_color(image):
    opencv_img = np.array(image.convert("RGB"))
    hsv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2HSV)
    avg_brightness = np.mean(hsv_img[:, :, 2])
    color_hue = np.mean(hsv_img[:, :, 0])

    color = "Brown" if (color_hue < 30 or color_hue > 150) else "White"
    cleanliness = "Dirty" if avg_brightness < 100 else "Clean"
    return color, cleanliness

def detect_size(image):
    opencv_img = np.array(image.convert("RGB"))
    blurred = cv2.GaussianBlur(opencv_img, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        _, _, w, _ = cv2.boundingRect(largest)
        return "Large" if w >= 100 else "Medium" if w >= 70 else "Small"
    return "Unknown"

def generate_heatmap(model, img_array, original_image, layer_name=None):
    """
    Builds the model (and any nested sub‑model), finds the last conv layer
    inside that sub‑model, then computes and superimposes Grad‑CAM.
    """
    try:
        # 1) Unwrap a nested Sequential if it exists
        target_model = model
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                target_model = layer
                break

        # 2) Call both models to ensure their inputs/outputs are built
        _ = model(img_array, training=False)
        _ = target_model(img_array, training=False)

        # 3) Auto‑detect last conv layer if none specified
        if layer_name is None:
            for layer in reversed(target_model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
            if layer_name is None:
                print("No conv layer found in nested model.")
                return None

        # 4) Build grad_model: inputs from nested, outputs from nested + top‑level
        grad_model = tf.keras.models.Model(
            [target_model.input],
            [ target_model.get_layer(layer_name).output,
              model.output ]  # use top‑level for final predictions
        )

        # 5) Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)

        # 6) Superimpose onto original
        heatmap = cv2.resize(heatmap.numpy(), (original_image.width, original_image.height))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(np.array(original_image), 0.6, heatmap, 0.4, 0)
        return Image.fromarray(superimposed)

    except Exception as e:
        print(f"Heatmap error: {e}")
        # fallback to blank image
        return Image.fromarray(np.zeros_like(np.array(original_image)))


# ------------------ Main Egg Analyzer ------------------

def analyze_egg(image, threshold=0.5):
    if image is None:
        return None, None, "No image received."

    original = image.copy()
    resized = original.resize((128, 128))
    img_array = np.expand_dims(np.array(resized) / 255.0, axis=0)

    try:
        score = model.predict(img_array, verbose=0)[0][0]
    except Exception as e:
        return None, None, f"Prediction error: {str(e)}"

    crack_result = "Defective (Cracked)" if score >= threshold else "Not Defective"
    color, cleanliness = detect_cleanliness_and_color(original)
    size_text = detect_size(original)
    gradient_img = generate_heatmap(model, img_array, original)

    result_text = f"""
    ### 🥚 Egg Inspection Result:
    - 🟠 Color: {color}
    - ✨ Cleanliness: {cleanliness}
    - ✅ Size: {size_text}
    - ❗ Defect Status: {crack_result}
    - 🔍 Confidence Score: {score:.2f}
    """

    draw = ImageDraw.Draw(original)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(20, original.width // 18))
    except:
        font = ImageFont.load_default()

    overlay = f"{color} | {cleanliness} | {size_text} | {crack_result}"
    draw.text((10, 10), overlay, fill=(255, 0, 0) if "Defective" in crack_result else (0, 128, 0), font=font)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] → {overlay} | Score: {score:.2f}\n")

    frame_history.append((time.time(), original))
    while frame_history and frame_history[0][0] < time.time() - 60:
        frame_history.popleft()

    return original, gradient_img, result_text

# ------------------ UI & Gradio Interface ------------------

def read_log():
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            return f.read()
    return "No logs yet."

with gr.Blocks() as demo:
    gr.Markdown("## 🥚 Smart Egg Analyzer (Single Image Mode)")

    with gr.Row():
        upload_input = gr.Image(label="Upload Egg Image", sources="upload", type="pil")
        threshold_slider = gr.Slider(0.1, 0.9, value=0.5, step=0.01, label="Defect Detection Threshold")

    analyze_button = gr.Button("🔍 Analyze Image")

    with gr.Row():
        annotated_output = gr.Image(label="🖼 Annotated Egg")
        gradient_output = gr.Image(label="🧠 Heatmap Visualization")
        analysis_output = gr.Markdown()

    analyze_button.click(
        analyze_egg,
        inputs=[upload_input, threshold_slider],
        outputs=[annotated_output, gradient_output, analysis_output]
    )

    gr.Markdown("### 📜 Prediction Log")
    log_box = gr.Textbox(label="🧾 Log", lines=8, interactive=False)
    refresh_btn = gr.Button("🔄 Refresh Log")
    gr.DownloadButton(label="📂 Download Log", value=log_file)

    gallery = gr.Gallery(label="📸 Recent Analysis History")
    refresh_gallery_btn = gr.Button("🔄 Refresh History")
    refresh_gallery_btn.click(lambda: [img for _, img in list(frame_history)], outputs=gallery)
    refresh_btn.click(read_log, outputs=log_box)

demo.launch()
