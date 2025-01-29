import cv2
import numpy as np
import torch

class Preprocessing:
    def __init__(self):
        self.clahe_clip_limit = 1.5
        self.clahe_tile_grid_size = (4, 4)
        self.dark_channel_window = 7
        self.guided_filter_radius = 7
        self.gamma = 1.2
        self.contrast_alpha = 1.3
        self.contrast_beta = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def dark_channel_prior(self, image):
        """Computes the dark channel of the image."""
        min_channel = np.min(image, axis=2)
        kernel = np.ones((self.dark_channel_window, self.dark_channel_window), np.uint8)
        dark_channel = cv2.erode(min_channel, kernel)
        return dark_channel

    def atmospheric_light(self, image, dark_channel, top_percent=0.001):
        """Estimates atmospheric light using the brightest pixels in the dark channel."""
        h, w = dark_channel.shape
        num_pixels = max(int(h * w * top_percent), 1)
        dark_flat = dark_channel.ravel()
        image_flat = image.reshape(-1, 3)
        indices = np.argsort(dark_flat)[-num_pixels:]
        brightest_pixels = image_flat[indices]
        A = np.mean(brightest_pixels, axis=0)
        return A

    def estimate_transmission(self, image, A, omega=0.95):
        """Estimates the transmission map based on atmospheric light and dark channel prior."""
        normalized_image = image.astype(np.float32) / A
        dark_channel = self.dark_channel_prior(normalized_image)
        transmission_estimate = 1 - omega * dark_channel
        return transmission_estimate

    def guided_filter(self, I, p, eps=1e-3):
        """Performs guided filtering for better transmission refinement."""
        kernel_size = 2 * (self.guided_filter_radius // 2) + 1
        mean_I = cv2.GaussianBlur(I, (kernel_size, kernel_size), 0)
        mean_p = cv2.GaussianBlur(p, (kernel_size, kernel_size), 0)
        corr_I = cv2.GaussianBlur(I * I, (kernel_size, kernel_size), 0)
        corr_Ip = cv2.GaussianBlur(I * p, (kernel_size, kernel_size), 0)
        var_I = corr_I - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_p
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        mean_a = cv2.GaussianBlur(a, (kernel_size, kernel_size), 0)
        mean_b = cv2.GaussianBlur(b, (kernel_size, kernel_size), 0)
        q = mean_a * I + mean_b
        return q

    def recover_scene(self, image, transmission, A, t_min=0.2):
        """Recovers the dehazed image using the estimated transmission and atmospheric light."""
        transmission = np.clip(transmission, t_min, 1)
        transmission = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)
        image_float = image.astype(np.float32)
        J = (image_float - A) / transmission + A
        J = np.clip(J, 0, 255).astype(np.uint8)
        return self.apply_gamma_correction(J)

    def apply_gamma_correction(self, image):
        """Applies gamma correction to improve brightness and contrast."""
        inv_gamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def preprocess_frame(self, frame):
        """Preprocesses the video frame using dehazing and contrast enhancement."""
        frame = cv2.convertScaleAbs(frame, alpha=self.contrast_alpha, beta=self.contrast_beta)
        dark_channel = self.dark_channel_prior(frame)
        A = self.atmospheric_light(frame, dark_channel)
        transmission = self.estimate_transmission(frame, A)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        refined_transmission = self.guided_filter(gray_frame, transmission)
        dehazed_frame = self.recover_scene(frame, refined_transmission, A)

        # Apply CLAHE for further enhancement
        lab = cv2.cvtColor(dehazed_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid_size)
        l = clahe.apply(l)
        final_frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        return final_frame

    def draw_filtered_detections(self, frame, detections, confidence_threshold=0.5):
        """Draws filtered detections on the frame with bounding boxes and confidence scores."""
        for detection in detections:
            x, y, w, h, confidence, class_id = detection
            if confidence >= confidence_threshold:
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Display confidence score
                label = f"ID {class_id}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
