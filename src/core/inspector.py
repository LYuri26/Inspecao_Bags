import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    is_bag: bool
    has_defect: bool
    confidence: float
    bbox: List[float]
    defect_type: Optional[str] = None


def simple_classify(
    raw_results: List[Dict[str, Any]], min_confidence: float = 0.5
) -> List[DetectionResult]:
    """
    Classificação binária simplificada:
    - Identifica se é sacola
    - Identifica se tem defeito (qualquer um)

    Args:
        raw_results: Resultados brutos do YOLO
        min_confidence: Confiança mínima para considerar detecção

    Returns:
        Lista de DetectionResult simplificados
    """
    results = []

    for detection in raw_results:
        try:
            class_name = detection.get("class_name", "").lower()
            confidence = float(detection.get("confidence", 0))

            if confidence < min_confidence:
                continue

            # Classificação binária
            is_bag = class_name == "sacola"
            has_defect = not is_bag  # Qualquer outra classe é considerada defeito

            results.append(
                DetectionResult(
                    is_bag=is_bag,
                    has_defect=has_defect,
                    confidence=confidence,
                    bbox=detection.get("bbox", []),
                    defect_type=class_name if has_defect else None,
                )
            )

        except Exception as e:
            logger.error(f"Erro ao classificar: {e}")

    return results


def validate_detections(
    detections: List[DetectionResult],
    bag_min_confidence: float = 0.4,
    defect_min_confidence: float = 0.4,
) -> Dict[str, any]:
    """
    Validação simplificada:
    - Sacola válida se confiança >= bag_min_confidence
    - Defeito válido se confiança >= defect_min_confidence

    Returns:
        {
            "valid_bag": bool,
            "has_defects": bool,
            "defects": List[str],  # Tipos de defeitos encontrados
            "main_confidence": float  # Confiança da detecção principal
        }
    """
    valid_bags = [
        d for d in detections if d.is_bag and d.confidence >= bag_min_confidence
    ]
    valid_defects = [
        d for d in detections if d.has_defect and d.confidence >= defect_min_confidence
    ]

    return {
        "valid_bag": len(valid_bags) > 0,
        "has_defects": len(valid_defects) > 0,
        "defects": list(set(d.defect_type for d in valid_defects)),
        "main_confidence": max([d.confidence for d in detections], default=0),
    }
