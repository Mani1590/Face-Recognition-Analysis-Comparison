import cv2
import numpy as np

class ImageAnalyzer:
    def analyze_image(self, image):
        """Perform comprehensive image analysis"""
        # Convert PIL Image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        results = {
            'blur_metrics': {
                'laplacian_var': self.measure_blur(cv_image),
                'is_blurry': self.is_image_blurry(gray)
            },
            'quality_metrics': {
                'overall_score': self.assess_quality(cv_image),
                'contrast': self.measure_contrast(gray),
                'brightness': self.measure_brightness(gray)
            },
            'resolution_metrics': {
                'dimensions': f"{image.size[0]}x{image.size[1]}",
                'megapixels': round((image.size[0] * image.size[1]) / 1000000, 2),
                'aspect_ratio': round(image.size[0] / image.size[1], 2)
            },
            'noise_metrics': {
                'noise_level': self.measure_noise(cv_image),
                'signal_to_noise': self.calculate_snr(gray)
            }
        }
        return results

    def measure_blur(self, image):
        """Measure image blur using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def is_image_blurry(self, gray_image, threshold=100.0):
        """Determine if image is blurry"""
        variance = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        return variance < threshold

    def assess_quality(self, image):
        """Overall image quality assessment"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = self.measure_contrast(gray)
        brightness = self.measure_brightness(gray)
        sharpness = self.measure_blur(image)
        
        quality_score = (
            (contrast / 127.0) * 0.4 +
            (brightness / 255.0) * 0.3 +
            (min(sharpness / 500.0, 1.0)) * 0.3
        ) * 100
        
        return round(quality_score, 2)

    def measure_contrast(self, gray_image):
        """Measure image contrast"""
        return float(cv2.meanStdDev(gray_image)[1][0][0])

    def measure_brightness(self, gray_image):
        """Measure image brightness"""
        return float(np.mean(gray_image))

    def measure_noise(self, image):
        """Measure image noise"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise_sigma = np.std(gray) / 255.0
        return round(noise_sigma * 100, 2)

    def calculate_snr(self, gray_image):
        """Calculate Signal-to-Noise Ratio"""
        mean_signal = np.mean(gray_image)
        std_noise = np.std(gray_image)
        
        if std_noise == 0:
            return float('inf')
            
        snr = 20 * np.log10(mean_signal / std_noise)
        return round(snr, 2)