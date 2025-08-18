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
from .utils import update_status, save_defect_image


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
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.base_path = os.path.join(BASE_DIR, "cadastros")
        self.bag_counter = 0
        self.last_bag_seen_time = time.time()
        self.current_bag_defects = []
        self.current_image = None

    def handle_new_frame(self, camera_id, frame):
        """Processa novos frames recebidos do CameraManager"""
        if frame is None:
            return

        try:
            # Usa a função unificada de processamento
            self.current_image = process_detection(self, frame, camera_id)
        except Exception as e:
            logger.error(
                f"Error processing camera {camera_id} frame: {str(e)}", exc_info=True
            )
            self.current_image = frame

    def update_frame(self):
        """Atualiza exibição das câmeras com detecção em tempo real"""
        frames = []

        for i in range(9):
            frame = self.camera_manager.get_latest_frame(i)
            if frame is None:
                frames.append(None)
                continue

            # Processa detecção em todos os frames
            detected_frame = process_detection(self, frame, i)

            # Gera preview reduzido para exibição
            h, w = detected_frame.shape[:2]
            scale = 640 / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            preview_frame = cv2.resize(detected_frame, (new_w, new_h))
            frames.append(preview_frame)

        # exibe no grid (3x3)
        self.update_grid_display(frames)

    def _handle_defect_detection(self, camera_id, defect_name, bbox, frame):
        """Handle detection of a specific defect"""
        # Verifica política da empresa antes de processar
        policy = self.active_company.get("policy", {}) if self.active_company else {}
        defect_key = self.defect_mapping.get(defect_name.lower())

        if defect_key and policy.get(defect_key, False):
            logger.debug(f"Defeito {defect_name} aceito pela política - ignorando")
            return

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

            # Save defect image only if not accepted by policy
            if not (defect_key and policy.get(defect_key, False)):
                self.save_defect_image(frame, defect_name.lower(), camera_id)

    def _save_defect_image(self, frame, defect_type, camera_id):
        """Salva imagem anotada com defeito no formato compatível com reports.py"""
        if not self.active_company:
            return None

        try:
            # Garante que a pasta existe
            reports_dir = self._ensure_reports_folder()
            if not reports_dir:
                return None

            # Pasta do dia atual
            day_folder = reports_dir / datetime.now().strftime("%d-%m-%Y")
            day_folder.mkdir(exist_ok=True)

            # Nome do arquivo no formato: timestamp-bagX-defeito-cameraY.jpg
            timestamp = datetime.now().strftime("%H-%M-%S")
            filename = f"{timestamp}-bag{self.bag_counter}-{defect_type}-camera{camera_id+1}.jpg"
            filepath = day_folder / filename

            # Salva a imagem
            cv2.imwrite(str(filepath), frame)
            logger.info(f"Imagem de defeito salva em: {filepath}")

            # Cria/atualiza arquivo de log
            log_file = day_folder / "defects_log.txt"
            with open(log_file, "a", encoding="utf-8") as f:
                log_entry = (
                    f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | "
                    f"Empresa: {self.active_company['name']} | "
                    f"Câmera: {camera_id+1} | "
                    f"Defeito: {defect_type} | "
                    f"Sacola: {self.bag_counter} | "
                    f"Arquivo: {filename}\n"
                )
                f.write(log_entry)

            return str(filepath)

        except Exception as e:
            logger.error(f"Erro ao salvar imagem do defeito: {str(e)}", exc_info=True)
            return None

    def _check_bag_timeout(self):
        """Check if bag needs to be changed due to timeout"""
        if time.time() - self.last_bag_seen_time > 15:
            self.bag_counter += 1
            self.current_bag_defects.clear()
            self.last_bag_seen_time = time.time()

    def update_frame(self):
        """Atualiza exibição das câmeras com detecção em tempo real"""
        frames = []

        for i in range(9):
            frame = self.camera_manager.get_latest_frame(i)
            if frame is None:
                frames.append(None)
                continue

            # Processa detecção em todos os frames
            detected_frame = process_detection(self, frame, i)

            # Gera preview reduzido para exibição
            h, w = detected_frame.shape[:2]
            scale = 640 / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            preview_frame = cv2.resize(detected_frame, (new_w, new_h))
            frames.append(preview_frame)

        # exibe no grid (3x3)
        self.update_grid_display(frames)

    def start_camera(self, camera_id=0):
        """Start capture from a specific camera"""
        if camera_id is None:
            self.start_all_cameras()
            return

        # Garante que a pasta reports existe antes de iniciar
        self._ensure_reports_folder()

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

    def _ensure_reports_folder(self):
        """Garante que a estrutura de pastas reports existe para a empresa ativa"""
        if not self.active_company:
            return None

        try:
            company_name = self.active_company["name"]
            safe_name = "".join(
                c for c in company_name if c.isalnum() or c in (" ", "-", "_")
            ).rstrip()

            # Navegar até a raiz do projeto
            base_dir = Path(__file__).resolve()
            while base_dir.name != "Inspecao_Bags":
                base_dir = base_dir.parent

            # Criar pasta reports se não existir
            reports_dir = base_dir / "cadastros" / safe_name / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            return reports_dir

        except Exception as e:
            logger.error(f"Erro ao criar pasta de reports: {str(e)}", exc_info=True)
            return None
