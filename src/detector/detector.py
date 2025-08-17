import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Union, List, Dict, Any
from ultralytics import YOLO
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BagDetector:
    """
    Detector de sacolas e defeitos associados (multiclasse).
    """

    CLASS_COLORS = {
        "Sacola": (0, 255, 0),  # Verde
        "rasgo": (255, 165, 0),  # Laranja
        "corte": (0, 0, 255),  # Vermelho
        "mancha": (255, 255, 0),  # Ciano
        "descostura": (255, 0, 255),  # Magenta
        "sujeira": (0, 255, 255),  # Amarelo
    }

    def __init__(self, model_path: Union[str, Path], confidence_threshold: float = 0.5):
        """
        Inicializa o detector com modelo personalizado.

        Args:
            model_path: Caminho para o modelo .pt treinado
            confidence_threshold: Confiança mínima para considerar a detecção
        """
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: Union[str, Path]) -> YOLO:
        """Carrega o modelo YOLO personalizado"""
        try:
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability()
                if major >= 12:  # placas muito novas ainda sem suporte
                    device = "cpu"
                else:
                    device = "cuda"
            else:
                device = "cpu"
            model = YOLO(model_path).to(device)
            logger.info(f"✅ Modelo carregado em {device.upper()}")
            return model
        except Exception as e:
            logger.error(f"❌ Falha ao carregar modelo: {e}")
            raise

    def detect(self, image: Union[str, Path, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Executa detecção na imagem (todas as classes).

        Returns:
            Lista de dicionários com bbox, score e classe
        """
        try:
            img = (
                self._load_image(image)
                if isinstance(image, (str, Path))
                else image.copy()
            )
            results = self.model(img)
            result = results[0]

            detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                for box, score, cls_id in zip(
                    result.boxes.xyxy, result.boxes.conf, result.boxes.cls
                ):
                    score = float(score)
                    if score >= self.confidence_threshold:
                        class_id = int(cls_id)
                        class_name = self.model.names[class_id]
                        detection = {
                            "bbox": box.cpu().numpy().tolist(),
                            "confidence": score,
                            "class_id": class_id,
                            "class_name": class_name,
                        }
                        detections.append(detection)

            logger.debug(f"🔍 {len(detections)} objeto(s) detectado(s)")
            return detections

        except Exception as e:
            logger.error(f"❌ Erro na detecção: {e}")
            return []

    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Carrega imagem do disco"""
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {img_path}")
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Falha ao ler imagem: {img_path}")
        return img

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Processa frame da câmera e desenha detecções multiclasse
        """
        try:
            results = self.model(frame)
            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                return frame

            for box, score, cls_id in zip(
                result.boxes.xyxy, result.boxes.conf, result.boxes.cls
            ):
                score = float(score)
                if score >= self.confidence_threshold:
                    class_id = int(cls_id)
                    class_name = self.model.names[class_id]
                    color = self.CLASS_COLORS.get(
                        class_name, (255, 255, 255)
                    )  # Branco se desconhecido
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    label = f"{class_name} {score:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

            return frame

        except Exception as e:
            logger.error(f"❌ Erro ao processar frame: {e}")
            return frame
