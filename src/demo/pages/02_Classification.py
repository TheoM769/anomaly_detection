import streamlit as st
from PIL import Image

from core.services.image_processing import ImageProcessingService

st.set_page_config(
    page_title="Classification",
    page_icon="🏷️",
)

st.title("Image Classification")

classification_mapping = {
    "ViT-B": "vit_cls",
    "DINO-B": "dino_cls"
}

selected_model = st.radio(
    "Select a model for inference",
    ["ViT-B", "DINO-B"],
    key="cls_model_selection",
)

model_name = st.session_state.cls_model_selection
with st.spinner("Loading model..."):
    model = ImageProcessingService.get_processor(classification_mapping[model_name])

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    # Show image in left column
    with col1:
        st.image(image, caption="Input Image")

    # Run inference and show predictions in right column
    if st.button("Run Inference"):
        with st.spinner("Running inference..."):
            predictions, inference_time = model.predict_with_inference_time(image)

        st.success("Inference completed!")

        with col2:
            st.subheader("Predictions")
            st.write(f"{predictions}")
            st.write(f"Inference time: {inference_time:.4f} seconds")