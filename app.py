# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from modules.initial_analysis import ImageAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Face Recognition Analysis",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_histogram_plot(image, title):
    fig = px.histogram(image.ravel(), 
                      title=title,
                      labels={'value':'Pixel Value', 'count':'Frequency'},
                      template='plotly_dark')
    fig.update_layout(showlegend=False, height=300)
    return fig

def create_quality_radar(results):
    categories = ['Blur', 'Quality', 'Noise', 'Brightness', 'Contrast']
    values = [
        min(100, results['blur_metrics']['laplacian_var']/500*100),
        results['quality_metrics']['overall_score'],
        results['noise_metrics']['noise_level'],
        results['quality_metrics']['brightness']/255*100,
        results['quality_metrics']['contrast']/127*100
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Image Metrics'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title="Image Quality Metrics"
    )
    return fig

def main():
    st.title("Face Recognition Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload your image for analysis",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        # Load and resize image
        image = Image.open(uploaded_file)
        max_width = 400
        image.thumbnail((max_width, max_width))
        
        # Display image and perform initial analysis automatically
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", width=400)
        
        with col2:
            st.markdown("""
                <div style='background-color: #1e1e1e; padding: 20px; border-radius: 10px;'>
                    <h3 style='color: white;'>Image Information</h3>
                """, unsafe_allow_html=True)
            st.write(f"Size: {image.size}")
            st.write(f"Format: {image.format}")
            st.write(f"Mode: {image.mode}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Perform initial analysis automatically
        with st.spinner("Performing initial analysis..."):
            analyzer = ImageAnalyzer()
            results = analyzer.analyze_image(image)
            
            # Store results in session state
            st.session_state['initial_analysis'] = results
            
            # Display comprehensive analysis results
            st.subheader("Initial Analysis Results")
            
            # Quality metrics using radar chart
            st.plotly_chart(create_quality_radar(results), use_container_width=True)
            
            # Detailed metrics in columns
            col1, col2, col3 = st.columns(3)
            
            # Image histogram
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            with col1:
                st.plotly_chart(create_histogram_plot(gray, "Intensity Distribution"))
            
            with col2:
                st.metric("Quality Score", f"{results['quality_metrics']['overall_score']:.1f}%")
                st.metric("Blur Level", f"{min(100, results['blur_metrics']['laplacian_var']/500*100):.1f}%")
            
            with col3:
                st.metric("Signal-to-Noise Ratio", f"{results['noise_metrics']['signal_to_noise']:.1f} dB")
                st.metric("Brightness", f"{results['quality_metrics']['brightness']:.1f}")
            
            # Resolution details
            st.info(f"""
            📊 Resolution Analysis:
            • Dimensions: {results['resolution_metrics']['dimensions']}
            • Megapixels: {results['resolution_metrics']['megapixels']} MP
            • Aspect Ratio: {results['resolution_metrics']['aspect_ratio']}
            """)
            
            # Show "Start Detailed Analysis" button only after initial analysis
            if st.button("Start Detailed Analysis", key="detailed_analysis"):
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                
                tab1, tab2, tab3 = st.tabs([
                    "Eigenface Analysis",
                    "LBP Analysis",
                    "Deep Learning Analysis"
                ])
                
                with tab1:
                    st.header("Eigenface Analysis")
                    st.write("Analysis results will appear here")
                
                with tab2:
                    st.header("LBP Analysis")
                    st.write("Analysis results will appear here")
                
                with tab3:
                    st.header("Deep Learning Analysis")
                    st.write("Analysis results will appear here")

if __name__ == "__main__":
    main()
