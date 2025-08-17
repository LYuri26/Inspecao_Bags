from PyQt5.QtCore import pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import logging
import os
import time
from pathlib import Path
from datetime import datetime
from .detection_logic import process_detection, display_image
from .sound_handler import SoundHandler
from .camera_manager import CameraManager
from .ui_elements import BaseCameraUI
from .utils import update_status

logger = logging.getLogger(__name__)


class CameraView(BaseCameraUI):
    def __init__(self, parent=None, model=None):
        super().__init__(parent)
        self.model = model

        # Color settings and defect mapping
        self.class_colors = {
            "Sacola": (0, 255, 0),
            "Rasgo": (255, 165, 0),
            "Corte": (0, 0, 255),
            "Mancha": (255, 255, 0),
            "Descostura": (255, 0, 255),
            "Sujeira": (0, 255, 255),
        }
        self.defect_mapping = {
            "rasgo": "aceita_rasgos",
            "corte": "aceita_cortes",
            "mancha": "aceita_manchas",
            "descostura": "aceita_descosturas",
            "sujeira": "aceita_sujeiras",
        }

        # Camera manager for RTSP streams
        self.camera_manager = CameraManager(num_cameras=9)
        self.camera_manager.frame_ready.connect(self.handle_new_frame)

        # Alert configuration - REMOVED sound_cooldown_secs parameter
        self.sound_handler = SoundHandler(
            alert_layout=self.alert_layout,
            alert_panel=self.alert_panel,
            results_label=self.results_label,
            parent=self,
        )

        # Timer for UI updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Paths and counters
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.base_path = os.path.join(BASE_DIR, "cadastros")
        self.bag_counter = 0
        self.last_bag_seen_time = time.time()
        self.current_bag_defects = []  # list with (defect, x, y)
        self.current_image = None

    def handle_new_frame(self, camera_id, frame):
        """Process new frames received from CameraManager"""
        if frame is None:
            return

        try:
            processed_frame = frame.copy()

            # Check if model is loaded
            if not self.model or not hasattr(self.model, "model"):
                cv2.putText(
                    processed_frame,
                    "Modelo não carregado",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                self.current_image = processed_frame
                return

            # Apply detection to the frame
            results = self.model.model(frame)[0]
            if not getattr(results, "boxes", None) or len(results.boxes) == 0:
                self._check_bag_timeout()
                self.current_image = processed_frame
                return

            # Process detections
            policy = (
                self.active_company.get("policy", {}) if self.active_company else {}
            )
            detected_bag = False

            for box, score, cls_id in zip(
                results.boxes.xyxy, results.boxes.conf, results.boxes.cls
            ):
                score = float(score)
                if score < 0.6:  # Confidence threshold
                    continue

                class_name = self.model.model.names[int(cls_id)]
                name_l = class_name.lower()
                name_d = class_name.capitalize()

                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.cpu().numpy())

                # Update time if bag is detected
                if name_l == "sacola":
                    detected_bag = True
                    self.last_bag_seen_time = time.time()

                # Check company policy
                if name_l in self.defect_mapping and policy.get(
                    self.defect_mapping[name_l], False
                ):
                    continue

                # Draw bounding box
                color = self.class_colors.get(name_d, (255, 255, 255))
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{name_d} ({score:.2f})"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    processed_frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1
                )
                cv2.putText(
                    processed_frame,
                    label,
                    (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

                # Handle defects
                if name_l in self.defect_mapping and not policy.get(
                    self.defect_mapping[name_l], False
                ):
                    self._handle_defect_detection(
                        camera_id, name_d, (x1, y1, x2, y2), processed_frame
                    )

            # Check for bag change timeout
            if detected_bag and (time.time() - self.last_bag_seen_time > 15):
                self.bag_counter += 1
                self.current_bag_defects.clear()
                self.last_bag_seen_time = time.time()

            self.current_image = processed_frame

        except Exception as e:
            logger.error(
                f"Error processing camera {camera_id} frame: {str(e)}", exc_info=True
            )
            self.current_image = frame

    def _handle_defect_detection(self, camera_id, defect_name, bbox, frame):
        """Handle detection of a specific defect"""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Group by 20px blocks to avoid duplicate alerts
        pos_key = (cx // 20, cy // 20)

        if pos_key not in self.current_bag_defects:
            self.current_bag_defects.append(pos_key)

            # Trigger alert (single sound for all defects)
            self.sound_handler.trigger_alert(
                f"Defeito detectado: {defect_name} (Câmera {camera_id+1}, Sacola {self.bag_counter+1})"
            )

            # Save defect image
            self._save_defect_image(frame, defect_name.lower())

    def _save_defect_image(self, frame, defect_type):
        """Save image of detected defect"""
        if not self.active_company:
            return

        try:
            company_name = self.active_company["name"]
            safe_name = "".join(
                c for c in company_name if c.isalnum() or c in (" ", "-", "_")
            ).rstrip()

            base_dir = Path(__file__).resolve()
            while base_dir.name != "Inspecao_Bags":
                base_dir = base_dir.parent

            reports_dir = base_dir / "cadastros" / safe_name / "reports"
            day_folder = reports_dir / datetime.now().strftime("%d-%m-%Y")
            day_folder.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%H-%M-%S")
            filename = f"{timestamp}-bag{self.bag_counter+1}-{defect_type}.jpg"
            cv2.imwrite(str(day_folder / filename), frame)

        except Exception as e:
            logger.error(f"Error saving defect image: {str(e)}")

    def _check_bag_timeout(self):
        """Check if bag needs to be changed due to timeout"""
        if time.time() - self.last_bag_seen_time > 15:
            self.bag_counter += 1
            self.current_bag_defects.clear()
            self.last_bag_seen_time = time.time()

    def update_frame(self):
        """Update display with latest frames (3x3 grid)"""
        frames = []
        for i in range(9):
            frame = self.camera_manager.get_latest_frame(i)
            if frame is not None:
                frame = process_detection(self, frame)
            frames.append(frame)

        self.update_grid_display(frames)

    def start_camera(self, camera_id=0):
        """Start capture from a specific camera"""
        if camera_id is None:
            self.start_all_cameras()
            return

        if not self.camera_manager.running:
            self.camera_manager.start_capture()

        status = (
            f"Câmera {camera_id+1} ativa - Monitorando {self.active_company['name']}"
            if self.active_company
            else f"Câmera {camera_id+1} ativa"
        )
        update_status(self, status)
        self.camera_state_changed.emit(True)
        if not self.timer.isActive():
            self.timer.start(30)

    def start_all_cameras(self):
        """Start all available cameras"""
        for i in range(9):
            self.start_camera(i)

    def stop_camera(self):
        """Stop all cameras and release resources"""
        if hasattr(self, "camera_manager"):
            self.camera_manager.stop_capture()
        if self.timer.isActive():
            self.timer.stop()
        update_status(self, "Câmeras desativadas")
        self.camera_state_changed.emit(False)

    def activate_camera(self):
        """Activate camera (alias for start_camera)"""
        self.start_camera()

    def deactivate_camera(self):
        """Deactivate camera (alias for stop_camera)"""
        self.stop_camera()

    def display_placeholder(self, text="SEM IMAGEM"):
        """Display a placeholder image"""
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            text,
            (100, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        display_image(self, placeholder)

    def set_model(self, model):
        """Set the detection model to be used"""
        self.model = model

    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        self.update_frame()

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        super().mousePressEvent(event)
        # Add custom mouse handling if needed

    def cleanup(self):
        """Clean up resources"""
        self.stop_camera()
        if hasattr(self, "sound_handler"):
            self.sound_handler.cleanup()
