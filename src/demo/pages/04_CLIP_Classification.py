import streamlit as st
from PIL import Image
import torch
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE

from core.services.image_processing import ImageProcessingService

st.set_page_config(
    page_title="CLIP Classification",
    page_icon="🔗",
    layout="wide",
)

st.title("CLIP Classification")
st.markdown("Use CLIP model to classify images using text prompts")

# Load the CLIP model
with st.spinner("Loading CLIP model..."):
    clip_model = ImageProcessingService.get_processor("clip")

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Define default prompts for each tab
default_binary_prompts = ["a good product", "a defective product"]
default_multiclass_prompts = ["a good product", "a scratched product", "a dented product", "a contaminated product", "a broken product"]

def get_embeddings(image, prompts):
    """Get CLIP embeddings for both image and prompts"""
    with torch.no_grad():
        image_features = clip_model.clip_backbone.image_feature_extraction(image)
        text_features = clip_model.clip_backbone.text_feature_extraction(prompts)
    return image_features.cpu().numpy(), text_features.cpu().numpy()

def create_tsne_visualization(image, prompts):
    """Create t-SNE visualization of image and text embeddings"""
    # Get embeddings
    image_embedding, text_embeddings = get_embeddings(image, prompts)
    
    # Combine embeddings for t-SNE
    combined_embeddings = np.vstack([image_embedding, text_embeddings])
    
    # Apply t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(2, len(combined_embeddings)-1)))
    tsne_result = tsne.fit_transform(combined_embeddings)
    
    # Split back into image and text embeddings
    image_point = tsne_result[0:1]
    text_points = tsne_result[1:]
    
    # Create visualization
    fig = go.Figure()
    
    # Add text points
    fig.add_trace(go.Scatter(
        x=text_points[:, 0],
        y=text_points[:, 1],
        mode='markers+text',
        marker=dict(size=10, color='blue'),
        text=prompts,
        textposition="top center",
        name='Text Prompts'
    ))
    
    # Add image point
    fig.add_trace(go.Scatter(
        x=image_point[:, 0],
        y=image_point[:, 1],
        mode='markers',
        marker=dict(size=15, color='red', symbol='diamond'),
        name='Image'
    ))
    
    # Update layout
    fig.update_layout(
        title="t-SNE Visualization of CLIP Embeddings",
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Create layout with two columns
    col1, col2 = st.columns([1, 2])
    
    # Show image in left column
    with col1:
        st.image(image, caption="Input Image", use_container_width=True)
    
    # Create tabs for different types of prompts
    with col2:
        binary_tab, multiclass_tab, personalized_tab = st.tabs(["Binary Prompts", "Multiclass Prompts", "Personalized Prompts"])
        
        # Binary Prompts Tab
        with binary_tab:
            st.subheader("Binary Classification")
            st.write("Classify between two options")
            
            # Text inputs for binary prompts
            binary_prompt1 = st.text_input("Positive Prompt", value=default_binary_prompts[0], key="binary_prompt1")
            binary_prompt2 = st.text_input("Negative Prompt", value=default_binary_prompts[1], key="binary_prompt2")
            
            binary_prompts = [binary_prompt1, binary_prompt2]
            
            if st.button("Run Binary Classification"):
                with st.spinner("Running inference..."):
                    result, inference_time = clip_model.predict_with_inference_time(binary_prompts, image)
                
                results_tab, viz_tab = st.tabs(["Results", "Visualization"])
                
                with results_tab:
                    st.success("Classification complete!")
                    st.write(f"**Result:** {result.predicted_class}")
                    st.write(f"**Confidence Score:** {result.score:.4f}")
                    st.write(f"**Inference Time:** {inference_time:.4f} seconds")
                
                with viz_tab:
                    st.subheader("Embedding Visualization")
                    st.write("This visualization shows how the image embedding relates to the text prompt embeddings in 2D space.")
                    fig = create_tsne_visualization(image, binary_prompts)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Multiclass Prompts Tab
        with multiclass_tab:
            st.subheader("Multiclass Classification")
            st.write("Classify among multiple options")
            
            # Text area for multiclass prompts (one per line)
            multiclass_prompts_text = st.text_area(
                "Enter one prompt per line", 
                value="\n".join(default_multiclass_prompts),
                height=150,
                key="multiclass_prompts"
            )
            
            multiclass_prompts = [p.strip() for p in multiclass_prompts_text.split('\n') if p.strip()]
            
            if st.button("Run Multiclass Classification"):
                if len(multiclass_prompts) < 2:
                    st.error("Please enter at least 2 prompts for multiclass classification")
                else:
                    with st.spinner("Running inference..."):
                        result, inference_time = clip_model.predict_with_inference_time(multiclass_prompts, image)
                    
                    results_tab, viz_tab = st.tabs(["Results", "Visualization"])
                    
                    with results_tab:
                        st.success("Classification complete!")
                        st.write(f"**Result:** {result.predicted_class}")
                        st.write(f"**Confidence Score:** {result.score:.4f}")
                        st.write(f"**Inference Time:** {inference_time:.4f} seconds")
                    
                    with viz_tab:
                        st.subheader("Embedding Visualization")
                        st.write("This visualization shows how the image embedding relates to the text prompt embeddings in 2D space.")
                        fig = create_tsne_visualization(image, multiclass_prompts)
                        st.plotly_chart(fig, use_container_width=True)
        
        # Personalized Prompts Tab
        with personalized_tab:
            st.subheader("Personalized Template Classification")
            st.write("Use a template to generate classification prompts")
            
            # Template input
            template = st.text_input(
                "Template (use {} as placeholder for class names)", 
                value="a {} product",
                key="template"
            )
            
            # Class names input
            class_names_text = st.text_area(
                "Enter class names (one per line)",
                value="good\ndefective\nscratched\ndented",
                height=150,
                key="class_names"
            )
            
            class_names = [c.strip() for c in class_names_text.split('\n') if c.strip()]
            
            # Generate prompts using the template
            personalized_prompts = [template.format(name) for name in class_names]
            
            # Preview the generated prompts
            if personalized_prompts:
                st.write("Generated prompts:")
                for prompt in personalized_prompts:
                    st.write(f"- {prompt}")
            
            if st.button("Run Personalized Classification"):
                if len(personalized_prompts) < 2:
                    st.error("Please enter at least 2 class names for classification")
                elif '{}' not in template:
                    st.error("Template must contain {} as a placeholder for class names")
                else:
                    with st.spinner("Running inference..."):
                        result, inference_time = clip_model.predict_with_inference_time(personalized_prompts, image)
                    
                    # Extract the original class name from the result
                    for name, prompt in zip(class_names, personalized_prompts):
                        if prompt == result.predicted_class:
                            predicted_class = name
                            break
                    else:
                        predicted_class = result.predicted_class
                    
                    results_tab, viz_tab = st.tabs(["Results", "Visualization"])
                    
                    with results_tab:
                        st.success("Classification complete!")
                        st.write(f"**Result:** {predicted_class}")
                        st.write(f"**Full Prompt:** {result.predicted_class}")
                        st.write(f"**Confidence Score:** {result.score:.4f}")
                        st.write(f"**Inference Time:** {inference_time:.4f} seconds")
                    
                    with viz_tab:
                        st.subheader("Embedding Visualization")
                        st.write("This visualization shows how the image embedding relates to the text prompt embeddings in 2D space.")
                        fig = create_tsne_visualization(image, personalized_prompts)
                        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please upload an image to start classification")
