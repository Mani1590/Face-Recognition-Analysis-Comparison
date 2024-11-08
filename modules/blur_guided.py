import cv2
import numpy as np
from PIL import Image
import face_recognition
import streamlit as st
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from skimage import exposure

class BlurGuidedRecognition:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def convert_to_cv2(self, pil_image):
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def convert_to_pil(self, cv2_image):
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

    def analyze_image(self, image):
        cv2_image = self.convert_to_cv2(image)
        results = {
            'blur_score': self.calculate_blur_metric(cv2_image),
            'face_details': self.analyze_facial_features(cv2_image),
            'lighting_quality': self.analyze_lighting(cv2_image),
            'face_symmetry': self.analyze_face_symmetry(cv2_image),
            'image_enhancement': self.enhance_image(cv2_image)
        }
        return results

    def calculate_blur_metric(self, image):
        """Analysis 1: Blur Detection and Measurement"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize the blur score between 0 and 100
        blur_score = min(100, max(0, laplacian_var / 500 * 100))
        
        return {
            'score': blur_score,
            'is_blurry': blur_score < 50,
            'recommendation': 'Image is too blurry, consider retaking' if blur_score < 50 else 'Image clarity is acceptable'
        }

    def analyze_facial_features(self, image):
        """Analysis 2: Facial Feature Detection and Analysis"""
        face_locations = face_recognition.face_locations(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not face_locations:
            return {'faces_found': 0, 'details': None}

        face_landmarks = face_recognition.face_landmarks(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        features_analysis = {
            'faces_found': len(face_locations),
            'details': []
        }

        for landmarks in face_landmarks:
            features = {
                'eyes_detected': 'left_eye' in landmarks and 'right_eye' in landmarks,
                'nose_detected': 'nose_bridge' in landmarks,
                'mouth_detected': 'top_lip' in landmarks and 'bottom_lip' in landmarks,
                'feature_count': len(landmarks.keys())
            }
            features_analysis['details'].append(features)

        return features_analysis

    def analyze_lighting(self, image):
        """Analysis 3: Lighting Quality Assessment"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Calculate mean brightness and contrast
        mean_brightness = gray.mean()
        contrast = gray.std()
        
        # Calculate histogram uniformity
        uniformity = np.sum(hist ** 2)
        
        return {
            'mean_brightness': mean_brightness,
            'contrast': contrast,
            'uniformity': uniformity,
            'lighting_quality': 'Good' if (40 < mean_brightness < 220 and contrast > 40) else 'Poor',
            'recommendations': self.get_lighting_recommendations(mean_brightness, contrast)
        }

    def analyze_face_symmetry(self, image):
        """Analysis 4: Face Symmetry Analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return {'symmetry_score': 0, 'is_symmetric': False}
            
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            if face.shape[1] % 2 != 0:
                face = face[:, :-1]
            
            # Split face into left and right
            left_side = face[:, :face.shape[1]//2]
            right_side = face[:, face.shape[1]//2:]
            right_side = cv2.flip(right_side, 1)
            
            # Calculate symmetry using SSIM
            symmetry_score = ssim(left_side, right_side)
            
            return {
                'symmetry_score': symmetry_score * 100,
                'is_symmetric': symmetry_score > 0.75,
                'details': {
                    'left_right_difference': ((1 - symmetry_score) * 100),
                    'symmetry_quality': 'Good' if symmetry_score > 0.75 else 'Poor'
                }
            }

    def enhance_image(self, image):
        """Analysis 5: Image Enhancement and Quality Improvement"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge channels
        enhanced_lab = cv2.merge((cl,a,b))
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Calculate improvement metrics
        original_quality = self.calculate_image_quality(image)
        enhanced_quality = self.calculate_image_quality(enhanced_image)
        
        return {
            'enhanced_image': self.convert_to_pil(enhanced_image),
            'quality_improvement': {
                'original_quality': original_quality,
                'enhanced_quality': enhanced_quality,
                'improvement_percentage': ((enhanced_quality - original_quality) / original_quality) * 100
            }
        }

    def calculate_image_quality(self, image):
        """Helper function to calculate image quality"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def get_lighting_recommendations(self, brightness, contrast):
        """Helper function to generate lighting recommendations"""
        recommendations = []
        if brightness < 40:
            recommendations.append("Increase lighting in the environment")
        elif brightness > 220:
            recommendations.append("Reduce exposure or ambient lighting")
        
        if contrast < 40:
            recommendations.append("Improve lighting contrast")
        
        return recommendations if recommendations else ["Lighting conditions are optimal"]