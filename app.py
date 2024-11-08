import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set page config first, before any other Streamlit commands
st.set_page_config(
    page_title="Face Recognition Analysis",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    with open('C:\\Users\\91983\\OneDrive\\Desktop\\Ongoing Projects\\Face Recognition Project (DIP)\\assets\\css\\styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    # Load custom CSS
    load_css()
    
    # Custom header with styling
    st.markdown("""
        <div class="header">
            <h1>Face Recognition Analysis</h1>
            <p>Analyze face recognition techniques in low-quality blurred images</p>
        </div>
    """, unsafe_allow_html=True)

    # Upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload your image for analysis",
        type=['jpg', 'jpeg', 'png']
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Display image and info
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.markdown("""
                <div class="metric-card">
                    <h3>Image Information</h3>
                """, unsafe_allow_html=True)
            st.write(f"Size: {image.size}")
            st.write(f"Format: {image.format}")
            st.write(f"Mode: {image.mode}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis button
        if st.button("Start Analysis", key="analyze_btn"):
            with st.spinner("Analyzing image..."):
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                
                # Create tabs for different analyses
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
                
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()