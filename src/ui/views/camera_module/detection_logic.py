import cv2
import numpy as np
import logging
import time
from .utils import save_defect_image  # reutiliza função utilitária
from src.core.inspector import simple_classify, validate_detections

logger = logging.getLogger(__name__)


def process_detection(self, frame, camera_id=None):
    """
    Orquestra a detecção:
    - Chama BagDetector (YOLO)
    - Interpreta via Inspector
    - Aplica políticas e salva imagens
    - Dispara alertas
    - Retorna frame anotado
    """
    if frame is None or frame.size == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    annotated_frame = frame.copy()

    # 1. Detecção YOLO via BagDetector
    detections = self.model.detect(frame)

    if not detections:
        check_bag_timeout(self)
        return annotated_frame

    # 2. Interpretação via Inspector
    classified = simple_classify(detections, 2)
    validated = validate_detections(classified)

    # Atualiza estado da sacola
    if validated["valid_bag"]:
        self.last_bag_seen_time = time.time()

    # 3. Políticas de empresa
    policy = self.active_company.get("policy", {}) if self.active_company else {}

    for det in classified:
        if det.has_defect:
            defect_key = det.defect_type
            if policy.get(defect_key, False):  # ignorar se política libera
                continue

            # Evita duplicação do mesmo defeito na mesma sacola
            pos_key = (defect_key, det.bbox[0] // 20, det.bbox[1] // 20)
            if pos_key not in self.current_bag_defects:
                self.current_bag_defects.append(pos_key)

                alert_msg = (
                    f"Defeito detectado: {defect_key.capitalize()} "
                    f"(Câmera {camera_id+1 if camera_id is not None else 'N/A'}, "
                    f"Bag {self.bag_counter})"
                )
                self.sound_handler.trigger_alert(alert_msg, defect_key=defect_key)

                save_defect_image(self, annotated_frame, defect_key, camera_id)

    # 4. Desenho das detecções (sem rodar YOLO de novo)
    annotated_frame = self.model.draw_detections(annotated_frame, detections)

    # 5. Controle de timeout de sacola
    if time.time() - self.last_bag_seen_time > 15:
        self.bag_counter += 1
        self.current_bag_defects.clear()
        self.last_bag_seen_time = time.time()

    return annotated_frame


def check_bag_timeout(self):
    """Verifica e atualiza timeout da sacola"""
    if time.time() - self.last_bag_seen_time > 15:
        self.bag_counter += 1
        self.current_bag_defects.clear()
        self.last_bag_seen_time = time.time()
        logger.info(f"Troca automática para bag {self.bag_counter}")
