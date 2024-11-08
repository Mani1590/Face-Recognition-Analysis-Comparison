import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

class ImageAnalyzer:
    def analyze_image(self, image):
        # Convert PIL Image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        results = {
            'blur_level': self.measure_blur(cv_image),
            'quality_score': self.assess_quality(cv_image),
            'resolution': f"{image.size[0]}x{image.size[1]}",
            'noise_level': self.measure_noise(cv_image)
        }
        return results

    def measure_blur(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def assess_quality(self, image):
        # Implement quality assessment metrics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        quality_score = np.mean(gray) / 255.0
        return quality_score * 100

    def measure_noise(self, image):
        # Implement noise measurement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise_sigma = np.std(gray) / 255.0
        return noise_sigma * 100