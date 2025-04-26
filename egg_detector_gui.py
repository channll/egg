import gradio as gr
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load model (no changes here)
model = load_model(r"D:\vs code\py\app.py\egg_detector_model.h5")

def predict(image, threshold=0.6):
    # Preprocess image
    image = image.resize((256, 256))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction_score = model.predict(img_array)[0][0]  # Dynamically computed
    class_label = "Defective" if prediction_score > threshold else "Not Defective"
    return f"Score: {prediction_score:.4f} → {class_label}"

    # Save to file (appends each prediction)
    with open("results.txt", "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] Image Prediction: {result_str}\n")
    
    return result_str

# Gradio interface (unchanged)
gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil"),
        gr.Slider(0.1, 0.9, 0.6, label="Threshold")
    ],
    outputs="text",
    title="Egg Defect Detector"
).launch()
