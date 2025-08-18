import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


def process_detection(self, frame, camera_id=None):
    """
    Processa um frame para detecção de defeitos com:
    - Detecção em tempo real
    - Desenho de bounding boxes e labels
    - Registro de defeitos
    - Verificação de políticas da empresa
    - Tratamento robusto de erros
    """
    try:
        # Verifica se o frame é válido
        if frame is None or frame.size == 0:
            logger.warning("Frame vazio recebido para processamento")
            return np.zeros((480, 640, 3), dtype=np.uint8)

        annotated_frame = frame.copy()
        logger.debug(f"Iniciando processamento - Frame shape: {frame.shape}")

        # Verifica se o modelo está carregado
        if not self.model or not hasattr(self.model, "model"):
            cv2.putText(
                annotated_frame,
                "Modelo não carregado",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            logger.warning("Modelo YOLO não disponível para detecção")
            return annotated_frame

        # Executa a inferência do modelo
        try:
            results = self.model.model(frame)
            result = results[0]
        except Exception as e:
            logger.error(f"Erro na inferência do modelo: {str(e)}", exc_info=True)
            cv2.putText(
                annotated_frame,
                "Erro no modelo",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return annotated_frame

        # Verifica se há detecções
        if not getattr(result, "boxes", None) or len(result.boxes) == 0:
            logger.debug("Nenhuma detecção encontrada no frame")
            self._check_bag_timeout()
            return annotated_frame

        # Obtém política da empresa
        policy = self.active_company.get("policy", {}) if self.active_company else {}
        class_counts = {}
        detected_bag = False

        # Processa cada detecção
        for box, score, cls_id in zip(
            result.boxes.xyxy, result.boxes.conf, result.boxes.cls
        ):
            try:
                score = float(score)
                if score < 0.6:
                    continue

                class_name = self.model.model.names[int(cls_id)]
                name_l = class_name.lower()
                name_d = class_name.capitalize()

                # Coordenadas da bounding box
                x1, y1, x2, y2 = map(int, box.cpu().numpy())

                # Atualiza tempo se detectar sacola
                if name_l == "sacola":
                    detected_bag = True
                    self.last_bag_seen_time = time.time()

                # Verifica política da empresa
                if name_l in self.defect_mapping and policy.get(
                    self.defect_mapping[name_l], False
                ):
                    continue

                # Desenha a bounding box
                color = self.class_colors.get(name_d, (255, 255, 255))
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                label = f"{name_d} #{class_counts[class_name]} ({score:.2f})"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(
                    annotated_frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1
                )
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                )

                # Registra defeitos
                if name_l in self.defect_mapping and not policy.get(
                    self.defect_mapping[name_l], False
                ):
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    pos_key = (name_l, cx // 20, cy // 20)

                    if pos_key not in self.current_bag_defects:
                        self.current_bag_defects.append(pos_key)
                        alert_msg = f"Defeito detectado: {name_d} (Câmera {camera_id+1 if camera_id is not None else 'N/A'}, Bag {self.bag_counter})"
                        self.sound_handler.trigger_alert(alert_msg, defect_key=name_l)

                        # Salvamento robusto da imagem
                        if hasattr(self, "save_defect_image"):
                            self.save_defect_image(annotated_frame, name_l, camera_id)

            except Exception as e:
                logger.error(f"Erro ao processar detecção: {str(e)}", exc_info=True)
                continue

        # Troca de sacola se detectada mas com timeout
        if detected_bag and (time.time() - self.last_bag_seen_time > 15):
            self.bag_counter += 1
            self.current_bag_defects.clear()
            self.last_bag_seen_time = time.time()

        return annotated_frame

    except Exception as e:
        logger.error(f"Erro crítico: {str(e)}", exc_info=True)
        return frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)


def check_bag_timeout(self):
    """Verifica e atualiza timeout da sacola"""
    if time.time() - self.last_bag_seen_time > 15:
        self.bag_counter += 1
        self.current_bag_defects.clear()
        self.last_bag_seen_time = time.time()
        logger.info(f"Troca automática para bag {self.bag_counter}")


def display_image(self, image: np.ndarray):
    """Exibe imagem na interface"""
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtCore import Qt

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    q_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(q_img)
    self.camera_label.setPixmap(
        pixmap.scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
    )
