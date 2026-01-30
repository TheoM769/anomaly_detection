import streamlit as st

from adapters.config import load_processors
 
st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide"
)

@st.cache_resource
def init_models():
    load_processors()

init_models()

st.title("Anomaly Detection Application")
st.markdown("### Welcome to the Anomaly Detection Application")

st.write("""
This application provides tools for image analysis and anomaly detection.
Choose a page from the sidebar to get started.
""")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.header("🏷️ Classification")
    st.write("""
    Use the Classification page to classify anomalies in images.
    This tool helps identify what type of anomaly is present.
    """)
    st.page_link("pages/02_Classification.py", label="Go to Classification", icon="🏷️")

with col2:
    st.header("🔍 Object Detection")
    st.write("""
    Use the Object Detection page to locate and identify anomalies in images.
    This tool identifies where anomalies are located and what type they are.
    """)
    st.page_link("pages/03_Memory_bank_anomaly_detection.py", label="Go to Object Detection", icon="🔍")

with col3:
    st.header("🔗 CLIP Classification")
    st.write("""
    Use the CLIP Classification page for zero-shot classification.
    This tool allows you to classify images using custom text prompts.
    """)
    st.page_link("pages/04_CLIP_Classification.py", label="Go to CLIP Classification", icon="🔗")

with col4:
    st.header("🦉 OWLv2 Detection")
    st.write("""
    Use the OWLv2 Detection page for zero-shot object detection.
    This tool allows you to detect objects using text queries.
    """)
    st.page_link("pages/05_OWLv2_Detection.py", label="Go to OWLv2 Detection", icon="🦉")

st.markdown("---")

# Display sample images or information about the models
st.subheader("About the Models")
st.write("""
The application uses advanced machine learning models for image analysis:
- Classification models: ViT-B and DINO-B
- Object Detection models: YOLO and Faster R-CNN
- Zero-shot Classification: CLIP
- Zero-shot Object Detection: OWLv2
""")