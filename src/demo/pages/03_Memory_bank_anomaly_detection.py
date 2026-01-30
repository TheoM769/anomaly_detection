import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import torch
import numpy as np
from sklearn.manifold import TSNE

from core.services.image_processing import ImageProcessingService

st.set_page_config(
    page_title="Memory bank anomaly detection",
    page_icon="🔍",
    layout="wide",
)

st.title("Memory bank anomaly detection")

# Model mapping for object detection
od_mapping = {
    "Memory DINO": "memory_dino",
    "Memory ViT": "memory_vit",
}

selected_model = st.radio(
    "Select a model for object detection",
    ["Memory DINO", "Memory ViT"],
    key="od_model_selection",
)

# Add slider for in_bank_split parameter
in_bank_split = st.slider(
    "Memory bank split ratio",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.1,
    help="Controls what proportion of the memory bank to use (0 = none, 1 = all)"
)

model_name = st.session_state.od_model_selection
with st.spinner("Loading model and memory bank..."):
    model = ImageProcessingService.get_processor(od_mapping[model_name])
    print(model)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="anomaly_image")

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Create layout with two columns
    col1, col2 = st.columns([1, 1.5])
    
    # Show image in left column
    with col1:
        st.image(image, caption="Input Image", use_container_width=True)
    
    # Run inference button
    if st.button("Run Inference"):
        
        with st.spinner("Running anomaly detection..."):
            # Run model prediction
            detections, inference_time = model.predict_with_inference_time(image, in_bank_split=in_bank_split)
            memory_bank = model.get_memory_bank()
            
            # Extract features for the uploaded image
            with torch.no_grad():
                image_features = model.backbone.image_feature_extraction(image)
            
            # Calculate nearest neighbor distances for memory bank points
            memory_bank_distances, _ = model.knn.kneighbors(memory_bank)
            # Get second nearest neighbor (first is self) for memory bank points
            memory_bank_min_distances = memory_bank_distances[:, 1]
            
            # Calculate nearest neighbor distance for the uploaded image
            sample_distances, _ = model.knn.kneighbors(image_features)
            sample_min_distance = sample_distances[0, 0]  # First nearest neighbor
            
            # Get anomaly threshold used by the model (usually 95th percentile)
            anomaly_threshold = model.anomaly_threshold
            
            # t-SNE visualization processing
            # First, convert tensors to numpy if needed
            memory_bank_np = memory_bank.numpy() if isinstance(memory_bank, torch.Tensor) else memory_bank
            image_features_np = image_features.numpy() if isinstance(image_features, torch.Tensor) else image_features
            
            print(memory_bank_np.shape)
            print(image_features_np.shape)

            # Combine memory bank with new image features for t-SNE
            combined_data = np.vstack([memory_bank_np, image_features_np])
            
            # Perform t-SNE
            tsne = TSNE(
                n_components=3,
                perplexity=min(30, max(5, len(combined_data) // 5)),  # Adjust perplexity based on data size
                max_iter=1000,
                random_state=42
            )
            tsne_result = tsne.fit_transform(combined_data)
            
            # Split results back into memory bank and uploaded image
            memory_bank_tsne = tsne_result[:-1]  # All but last point
            uploaded_image_tsne = tsne_result[-1:]  # Just the last point
        
        # Display results in right column
        with col2:
            # Anomaly detection results
            st.subheader("Anomaly Detection Result")
            st.write(f"{detections}")
            st.write(f"Inference time: {inference_time:.4f} seconds")
            # Create a tab view for different visualizations
            viz_tab1, viz_tab2 = st.tabs(["t-SNE Visualization", "Distance Distribution"])
            
            with viz_tab1:
                # Create t-SNE visualization
                st.subheader("Memory Bank t-SNE Visualization")
                
                # Display t-SNE parameters
                perplexity = min(30, max(5, len(combined_data) // 5))
                st.caption(f"t-SNE parameters: perplexity={perplexity}, iterations=1000")
                
                # Create Plotly figure
                fig = go.Figure()
                
                # Add memory bank points
                fig.add_trace(go.Scatter3d(
                    x=memory_bank_tsne[:, 0],
                    y=memory_bank_tsne[:, 1],
                    z=memory_bank_tsne[:, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        opacity=0.7,
                    ),
                    name='Memory Bank Features'
                ))
                
                # Add uploaded image point as red
                fig.add_trace(go.Scatter3d(
                    x=uploaded_image_tsne[:, 0],
                    y=uploaded_image_tsne[:, 1],
                    z=uploaded_image_tsne[:, 2],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='red',
                        symbol='diamond',
                        opacity=1.0,
                    ),
                    name='Uploaded Image'
                ))
                
                # Update layout
                fig.update_layout(
                    title='3D t-SNE visualization of Memory Bank features',
                    scene=dict(
                        xaxis_title='t-SNE 1',
                        yaxis_title='t-SNE 2',
                        zaxis_title='t-SNE 3',
                    ),
                    height=500,
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                
                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
                
            with viz_tab2:
                st.subheader("Distance Distribution Histogram")
                
                # Create histogram with plotly
                hist_fig = go.Figure()
                
                # Add histogram of memory bank distances
                hist_fig.add_trace(go.Histogram(
                    x=memory_bank_min_distances,
                    nbinsx=30,
                    name='Train samples shortest distance to memory bank',
                    marker_color='blue',
                    opacity=0.6
                ))
                
                # Add vertical line for the uploaded image distance
                hist_fig.add_trace(go.Scatter(
                    x=[sample_min_distance, sample_min_distance],
                    y=[0, 50],  # Will be scaled automatically
                    mode='lines',
                    name='Current Sample shortest distance to memory bank',
                    line=dict(color='red', width=2, dash='solid')
                ))
                
                # Add vertical line for the anomaly threshold
                hist_fig.add_trace(go.Scatter(
                    x=[anomaly_threshold, anomaly_threshold],
                    y=[0, 50],  # Will be scaled automatically
                    mode='lines',
                    name='Anomaly Threshold',
                    line=dict(color='orange', width=2, dash='dash')
                ))
                
                # Update layout
                hist_fig.update_layout(
                    title='Distribution of Nearest Neighbor Distances',
                    xaxis_title='Distance to Nearest Neighbor',
                    yaxis_title='Count',
                    height=400,
                    bargap=0.01,
                    showlegend=True,
                    legend=dict(orientation='h', yanchor='top', y=-0.2),
                    margin=dict(l=0, r=0, b=80, t=40)
                )
                
                # Add text annotation for comparison
                st.plotly_chart(hist_fig, use_container_width=True)
                
                # Show exact distance values
                st.info(f"""
                    - Sample distance to nearest neighbor: {sample_min_distance:.4f}
                    - Anomaly threshold: {anomaly_threshold:.4f}
                    - {'**ANOMALY DETECTED**' if sample_min_distance > anomaly_threshold else 'Normal sample'} 
                    (Distance {'>' if sample_min_distance > anomaly_threshold else '<'} Threshold)
                """)
    else:
        # Display message when the inference has not been run yet
        with col2:
            st.info("Upload an image and click 'Run Inference' to see anomaly detection results and t-SNE visualization.")