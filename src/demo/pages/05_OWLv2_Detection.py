import streamlit as st
import torch
from PIL import Image
import numpy as np
import time
from adapters.backbones.owlv2 import Owlv2Backbone

st.set_page_config(
    page_title="OWLv2 Detection",
    page_icon="🦉",
    layout="wide"
)

st.title("OWLv2 Object Detection")
st.markdown("### Zero-shot Object Detection with OWLv2")

st.write("""
This page allows you to perform zero-shot object detection using the OWLv2 model.
Upload an image and provide text queries to detect objects in the image.
""")
@st.cache_resource
def init_model():
    model = Owlv2Backbone()
    model.load_weights()
    return model

model = init_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=400)

# Text input for queries
text_queries = st.text_input(
    "Enter object queries (comma-separated)",
    "person, car, dog",
    help="Enter the objects you want to detect, separated by commas"
)

# Add slider for topk
topk = st.slider(
    "Number of detections per query",
    min_value=1,
    max_value=10,
    value=2,
    help="Maximum number of detections to show for each query"
)

# Add inference button
run_inference = st.button("Run Detection", type="primary")

if uploaded_file is not None:
    
    if run_inference and text_queries:
        # Process text queries
        text_queries = [query.strip() for query in text_queries.split(",")]
        
        with st.spinner("Running detection..."):
            start_time = time.time()
            results = model.detect_objects(text_queries, image, topk=topk)
            image = model.visualize_results(text_queries, results, image)
            inference_time = time.time() - start_time
        
        # Display results
        st.subheader("Detection Results")
        st.info(f"Inference time: {inference_time:.2f} seconds")
        
        # Display the image with detections
        st.image(image, caption="Detection Results", width=600)
        
    elif run_inference and not text_queries:
        st.warning("Please enter text queries before running detection.") 