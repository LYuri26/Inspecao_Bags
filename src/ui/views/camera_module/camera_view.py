from PyQt5.QtCore import pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import logging
import os
import time
from pathlib import Path
from datetime import datetime
from .detection_logic import process_detection
from .utils import display_image
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
        self.current_images = {}  # guarda último frame processado de cada câmera
        # Alert configuration
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
        BASE_DIR = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "./../../../..")
        )
        self.base_path = os.path.join(BASE_DIR, "cadastros")
        self.bag_counter = 0
        self.last_bag_seen_time = time.time()
        self.current_bag_defects = []
        self.current_image = None

    def handle_new_frame(self, camera_id, frame):
        try:
            detected_frame = process_detection(self, frame, camera_id=camera_id)
            self.current_images[camera_id] = detected_frame
        except Exception as e:
            update_status(
                self, f"❌ Erro ao processar frame da câmera {camera_id+1}: {e}"
            )

    def update_frame(self):
        """Atualiza exibição das câmeras usando último frame processado de cada câmera"""
        frames = []

        for i in range(9):
            frame = self.camera_manager.get_latest_frame(i)
            if frame is None:
                frames.append(None)
                continue

            # ✅ Pega frame processado da câmera correspondente (se existir)
            detected_frame = self.current_images.get(i, frame)

            # Gera preview reduzido para exibição
            h, w = detected_frame.shape[:2]
            scale = 640 / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            preview_frame = cv2.resize(detected_frame, (new_w, new_h))
            frames.append(preview_frame)

        # Exibe no grid (3x3)
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
