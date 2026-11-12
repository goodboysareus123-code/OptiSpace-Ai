import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import pandas as pd

st.set_page_config(
    page_title="OptiSpace AI",
    page_icon="📐",
    layout="wide"
)

st.title("📐 OptiSpace AI: Room Layout Optimizer")
st.write("AI-powered furniture detection and room space analysis.")

@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

with st.spinner("Loading AI model..."):
    model = load_yolo_model()

INDOOR_CLASSES = {
    56: "Chair",
    57: "Couch",
    58: "Potted Plant",
    59: "Bed",
    60: "Dining Table",
    62: "TV/Monitor"
}

def analyze_spatial_layout(objects, image_width):
    suggestions = []
    left_mass = 0
    right_mass = 0

    for obj in objects:
        box = obj["box"]
        area = (box[2] - box[0]) * (box[3] - box[1])
        center_x = (box[0] + box[2]) / 2

        if center_x < image_width / 2:
            left_mass += area
        else:
            right_mass += area

    total_mass = left_mass + right_mass

    if total_mass > 0:
        imbalance = abs(left_mass - right_mass) / total_mass

        if imbalance > 0.4:
            heavy_side = "left" if left_mass > right_mass else "right"
            suggestions.append(
                f"Visual imbalance detected: More furniture area appears "
                f"on the {heavy_side} side. Consider rearranging furniture "
                f"for a more balanced layout."
            )
        else:
            suggestions.append(
                "Furniture appears reasonably balanced between the "
                "left and right sides."
            )

    return suggestions

uploaded_file = st.file_uploader(
    "Upload a room photograph",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.subheader("Your Room")
    st.image(image, use_container_width=True)

    with st.spinner("AI is analysing your room..."):
        results = model.predict(
            source=img_np,
            conf=0.35,
            verbose=False
        )

        detected_objects = []
        annotated = img_np.copy()

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])

                if cls_id in INDOOR_CLASSES:
                    coordinates = box.xyxy[0].cpu().numpy().tolist()
                    confidence = float(box.conf[0])
                    label = INDOOR_CLASSES[cls_id]

                    detected_objects.append({
                        "Object": label,
                        "Confidence": round(confidence * 100, 2),
                        "box": coordinates
                    })

                    x1, y1, x2, y2 = map(int, coordinates)

                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        3
                    )

                    cv2.putText(
                        annotated,
                        f"{label} {confidence:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

    st.success("Room analysis completed!")

    st.subheader("🤖 AI Object Detection")
    st.image(
        annotated,
        caption="Furniture detected by YOLOv8",
        channels="RGB",
        use_container_width=True
    )

    st.subheader("🪑 Detected Furniture")

    if detected_objects:
        display_data = [
            {
                "Object": obj["Object"],
                "Confidence": f'{obj["Confidence"]}%'
            }
            for obj in detected_objects
        ]

        st.dataframe(
            pd.DataFrame(display_data),
            use_container_width=True,
            hide_index=True
        )

        st.metric("Total Objects Detected", len(detected_objects))
    else:
        st.warning(
            "No supported furniture objects were detected. "
            "Try a clearer room photograph."
        )

    st.subheader("📐 Space Optimisation")

    suggestions = analyze_spatial_layout(
        detected_objects,
        img_np.shape[1]
    )

    if suggestions:
        for suggestion in suggestions:
            st.info(suggestion)

    st.caption(
        "Note: This version analyses visual furniture distribution. "
        "It does not calculate exact room dimensions or generate "
        "a new furniture layout."
    )
else:
    st.info("Upload a room photograph to begin AI analysis.")
