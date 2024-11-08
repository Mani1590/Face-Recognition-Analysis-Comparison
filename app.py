import streamlit as st
import cv2
import numpy as np
from PIL import Image
from modules.initial_analysis import ImageAnalyzer
from modules.recognition_techniques import (
    EigenfaceRecognition,
    LBPRecognition,
    DeepLearningRecognition
)
from modules.visualization import ResultsVisualizer

# Page configuration
st.set_page_config(
    page_title="Face Recognition Analysis",
    page_icon="👤",
    layout="wide"
)

def main():
    st.title("Face Recognition Analysis in Low-Quality Blurred Images")
    st.write("Upload an image to analyze using three different face recognition techniques")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Initial Analysis
        with st.spinner("Performing initial analysis..."):
            analyzer = ImageAnalyzer()
            initial_results = analyzer.analyze_image(image)

        # Display initial analysis results
        st.subheader("Initial Analysis Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Blur Level", f"{initial_results['blur_level']:.2f}")
        with col2:
            st.metric("Image Quality", f"{initial_results['quality_score']:.2f}")
        with col3:
            st.metric("Resolution", f"{initial_results['resolution']}")
        with col4:
            st.metric("Noise Level", f"{initial_results['noise_level']:.2f}")

        # Create tabs for different techniques
        tab1, tab2, tab3 = st.tabs([
            "Eigenface Analysis",
            "LBP Analysis",
            "Deep Learning Analysis"
        ])

        # Perform analysis with each technique
        with tab1:
            eigenface = EigenfaceRecognition()
            eigenface_results = eigenface.analyze(image)
            visualizer = ResultsVisualizer()
            visualizer.display_eigenface_results(eigenface_results)

        with tab2:
            lbp = LBPRecognition()
            lbp_results = lbp.analyze(image)
            visualizer.display_lbp_results(lbp_results)

        with tab3:
            deep_learning = DeepLearningRecognition()
            deep_learning_results = deep_learning.analyze(image)
            visualizer.display_deep_learning_results(deep_learning_results)

        # Comparison Section
        st.subheader("Technique Comparison")
        visualizer.display_comparison(
            eigenface_results,
            lbp_results,
            deep_learning_results
        )

        # Export Results
        if st.button("Generate Report"):
            report = generate_report(
                initial_results,
                eigenface_results,
                lbp_results,
                deep_learning_results
            )
            st.download_button(
                label="Download Report",
                data=report,
                file_name="face_recognition_analysis_report.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()