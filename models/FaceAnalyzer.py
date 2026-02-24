import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.spatial.transform import Rotation
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class FaceAnalyzer:
    def __init__(self, image_path):
        """Initialize the FaceAnalyzer with an image path."""
        self.image_path = image_path
        self.original_image = self.load_image()
        self.image_height, self.image_width = self.original_image.shape[:2]
        self.face_data = []
        self.zones = self.define_zones()
        
        # Initialize OpenCV face detector (DNN-based, more accurate than Haar)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        # Eye cascades for landmark approximation
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

    def load_image(self):
        """Load and validate the input image."""
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(f"Could not load image from {self.image_path}")
        image = self.enhance_image(image)
        return image

    def enhance_image(self, image):
        """Enhance image brightness and contrast using CLAHE for better face detection."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        logger.info("Image enhanced with CLAHE for better face detection")
        return enhanced_image

    def define_zones(self):
        """Define zones based on the image width."""
        third_width = self.image_width // 3
        return {
            "left": (0, third_width),
            "center": (third_width, 2 * third_width),
            "right": (2 * third_width, self.image_width)
        }

    def detect_faces_opencv(self):
        """Detect faces using OpenCV Haar cascade (no external model files needed)."""
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with multiple scale factors for better detection
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        result = {}
        for idx, (x, y, w, h) in enumerate(faces):
            face_id = f"face_{idx + 1}"
            
            # Detect eyes within the face region for landmark approximation
            face_roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(face_roi_gray, 1.1, 3, minSize=(15, 15))
            
            # Approximate landmarks from face geometry and detected eyes
            landmarks = self._approximate_landmarks(x, y, w, h, eyes)
            
            result[face_id] = {
                "score": 1.0,  # Haar cascade doesn't give a confidence score
                "facial_area": [x, y, x + w, y + h],
                "landmarks": landmarks
            }
        
        return result

    def _approximate_landmarks(self, x, y, w, h, eyes):
        """Approximate facial landmarks from face bounding box and detected eyes."""
        # Default landmarks based on face geometry (proportional estimates)
        landmarks = {
            "left_eye": [x + int(w * 0.3), y + int(h * 0.35)],
            "right_eye": [x + int(w * 0.7), y + int(h * 0.35)],
            "nose": [x + int(w * 0.5), y + int(h * 0.55)],
            "mouth_left": [x + int(w * 0.35), y + int(h * 0.75)],
            "mouth_right": [x + int(w * 0.65), y + int(h * 0.75)]
        }
        
        # If we detected eyes, use their actual positions
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate (left eye first)
            sorted_eyes = sorted(eyes, key=lambda e: e[0])
            ex1, ey1, ew1, eh1 = sorted_eyes[0]
            ex2, ey2, ew2, eh2 = sorted_eyes[1]
            landmarks["left_eye"] = [x + ex1 + ew1 // 2, y + ey1 + eh1 // 2]
            landmarks["right_eye"] = [x + ex2 + ew2 // 2, y + ey2 + eh2 // 2]
        
        return landmarks

    def calculate_head_pose(self, landmarks):
        """Calculate head pose angles using PnP."""
        try:
            model_points = np.array([
                (0.0, 0.0, 0.0),          # Nose tip
                (0.0, -330.0, -65.0),     # Chin
                (-225.0, 170.0, -135.0),  # Left eye corner
                (225.0, 170.0, -135.0),   # Right eye corner
                (-150.0, -150.0, -125.0), # Left mouth corner
                (150.0, -150.0, -125.0)   # Right mouth corner
            ])

            image_points = [
                landmarks["nose"],
                landmarks.get("chin", landmarks["nose"]),
                landmarks["left_eye"],
                landmarks["right_eye"],
                landmarks.get("mouth_left", landmarks["left_eye"]),
                landmarks.get("mouth_right", landmarks["right_eye"])
            ]
            image_points = np.array(image_points, dtype="double")

            focal_length = self.image_width
            center = (self.image_width / 2, self.image_height / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            dist_coeffs = np.zeros((4, 1))

            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                raise ValueError("PnP solution failed.")

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            rotation = Rotation.from_matrix(rotation_matrix)
            euler_angles = rotation.as_euler("xyz", degrees=True)
            pitch, yaw, roll = euler_angles

            return {
                "pitch": round(pitch, 2),
                "yaw": round(yaw, 2),
                "roll": round(roll, 2),
                "confidence": 1.0
            }
        except Exception as e:
            logger.warning(f"PnP head pose calculation failed: {str(e)}")
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "confidence": 0.0}

    def detect_emotion(self, face_image):
        """Detect emotion with confidence score using simple heuristics.
        Falls back to basic analysis if the emotion model is unavailable."""
        try:
            # Try using the custom emotion detection module
            from . import face_detection as fd
            analysis = fd.analyze(
                face_image, actions="emotion", enforce_detection=False, silent=True, 
            )
            emotions = analysis[0]["emotion"]
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            return {"emotion": dominant_emotion[0], "confidence": round(dominant_emotion[1] / 100, 2)}
        except Exception as e:
            logger.warning(f"Emotion detection via model failed: {str(e)}, using brightness heuristic")
            # Fallback: simple brightness-based heuristic
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
            mean_brightness = np.mean(gray)
            if mean_brightness > 140:
                return {"emotion": "happy", "confidence": 0.5}
            elif mean_brightness > 100:
                return {"emotion": "neutral", "confidence": 0.6}
            else:
                return {"emotion": "neutral", "confidence": 0.4}

    def determine_zone(self, center_x):
        """Determine which zone the student is located in based on x-coordinate."""
        for zone, (start, end) in self.zones.items():
            if start <= center_x < end:
                return zone
        return "unknown"

    def analyze_faces(self):
        """Main method to detect and analyze faces in the image."""
        try:
            # Use OpenCV Haar cascade (works without external model files)
            faces = self.detect_faces_opencv()
            
            if not faces:
                logger.warning("No faces detected in the image. The image may be too dark, "
                              "blurry, or faces may not be clearly visible.")
                return 0
            
            logger.info(f"Detected {len(faces)} face(s) in the image")
            
            for face_id, face in faces.items():
                x1, y1, x2, y2 = face["facial_area"]
                face_image = self.original_image[y1:y2, x1:x2]

                pose = self.calculate_head_pose(face["landmarks"])
                emotion_data = self.detect_emotion(face_image)
                zone = self.determine_zone((x1 + x2) // 2)

                face_data = {
                    "face_id": face_id,
                    "position": {
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        "center_x": (x1 + x2) // 2,
                        "center_y": (y1 + y2) // 2
                    },
                    "zone": zone,
                    "pose": pose,
                    "emotion": emotion_data["emotion"],
                    "confidence": emotion_data["confidence"],
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                self.face_data.append(face_data)
                self.draw_face_analysis(face_data)

            return len(self.face_data)
        except Exception as e:
            logger.error(f"Face analysis failed: {str(e)}")
            return 0

    def draw_face_analysis(self, face_data):
        """Draw bounding box and pitch/yaw/roll scores on each detected face."""
        pos = face_data["position"]
        x1, y1, x2, y2 = pos["x1"], pos["y1"], pos["x2"], pos["y2"]
        pose = face_data["pose"]
        pitch, yaw, roll = pose["pitch"], pose["yaw"], pose["roll"]

        # Color based on head pose deviation (green = facing forward, red = looking away)
        max_dev = max(abs(yaw), abs(pitch))
        if max_dev < 15:
            box_color = (0, 200, 0)       # Green — attentive
        elif max_dev < 35:
            box_color = (0, 200, 255)     # Yellow/Orange — moderate
        else:
            box_color = (0, 0, 230)       # Red — looking away

        # Draw bounding box
        cv2.rectangle(self.original_image, (x1, y1), (x2, y2), box_color, 2)

        # Corner accents for a polished look
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        thick = 3
        cv2.line(self.original_image, (x1, y1), (x1 + corner_len, y1), box_color, thick)
        cv2.line(self.original_image, (x1, y1), (x1, y1 + corner_len), box_color, thick)
        cv2.line(self.original_image, (x2, y1), (x2 - corner_len, y1), box_color, thick)
        cv2.line(self.original_image, (x2, y1), (x2, y1 + corner_len), box_color, thick)
        cv2.line(self.original_image, (x1, y2), (x1 + corner_len, y2), box_color, thick)
        cv2.line(self.original_image, (x1, y2), (x1, y2 - corner_len), box_color, thick)
        cv2.line(self.original_image, (x2, y2), (x2 - corner_len, y2), box_color, thick)
        cv2.line(self.original_image, (x2, y2), (x2, y2 - corner_len), box_color, thick)

        # Prepare label lines
        label_lines = [
            f"P:{pitch:.1f}  Y:{yaw:.1f}  R:{roll:.1f}",
            f"{face_data['emotion']} ({face_data['confidence']:.0%})",
            f"Zone: {face_data['zone']}",
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48
        font_thickness = 1
        line_height = 18
        padding = 4

        # Draw labels above bounding box with a dark background strip
        total_label_height = len(label_lines) * line_height + padding * 2
        label_top = max(0, y1 - total_label_height)

        # Semi-transparent background for labels
        overlay = self.original_image.copy()
        cv2.rectangle(overlay, (x1, label_top), (x2, y1), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, self.original_image, 0.3, 0, self.original_image)

        for i, text in enumerate(label_lines):
            ty = label_top + padding + (i + 1) * line_height - 4
            cv2.putText(self.original_image, text, (x1 + padding, ty),
                        font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    def save_results(self, output_image_path, output_csv_path):
        """Save the annotated image and CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_image_path = output_image_path or f"face_analysis_{timestamp}.jpg"
        output_csv_path = output_csv_path or f"face_data_{timestamp}.csv"

        # Save annotated image
        cv2.imwrite(output_image_path, self.original_image)
        logger.info(f"Annotated image saved to {output_image_path}")

        # Save face data to CSV only if data exists
        if self.face_data:
            df = pd.json_normalize(self.face_data)
            df.to_csv(output_csv_path, index=False)
            logger.info(f"Face data saved to {output_csv_path}")
        else:
            logger.warning("No face data available to save to CSV. Saving a placeholder file with default values.")
            
            # Create a placeholder DataFrame with default values
            placeholder_columns = ["face_id", "zone", "pose.pitch", "pose.yaw", "pose.roll", "confidence", "emotion"]
            placeholder_data = [{
                "face_id": 0,
                "zone": "unknown",
                "pose.pitch": "None",
                "pose.yaw": "None",
                "pose.roll": "None",
                "confidence": "None",
                "emotion": "None"
            }]
            placeholder_df = pd.DataFrame(placeholder_data, columns=placeholder_columns)
            placeholder_df.to_csv(output_csv_path, index=False)
            logger.info(f"Placeholder face data saved to {output_csv_path}")