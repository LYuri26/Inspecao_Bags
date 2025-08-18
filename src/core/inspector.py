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
    Interpreta resultados brutos do BagDetector.detect().
    - Identifica sacola
    - Identifica defeito

    Args:
        raw_results: Saída de BagDetector.detect()
        min_confidence: Confiança mínima para considerar detecção
    """
    results: List[DetectionResult] = []

    for det in raw_results:
        try:
            class_name = det["class_name"].lower()
            confidence = float(det["confidence"])
            bbox = det["bbox"]

            if confidence < min_confidence:
                continue

            is_bag = class_name == "sacola"
            has_defect = not is_bag

            results.append(
                DetectionResult(
                    is_bag=is_bag,
                    has_defect=has_defect,
                    confidence=confidence,
                    bbox=bbox,
                    defect_type=class_name if has_defect else None,
                )
            )
        except Exception as e:
            logger.error(f"Erro ao classificar: {e}")

    return results


def validate_detections(
    detections: List[DetectionResult],
    bag_min_confidence: float = 0.5,
    defect_min_confidence: float = 0.5,
) -> Dict[str, Any]:
    """
    Validações em cima dos DetectionResults.
    """
    valid_bags = [
        d for d in detections if d.is_bag and d.confidence >= bag_min_confidence
    ]
    valid_defects = [
        d for d in detections if d.has_defect and d.confidence >= defect_min_confidence
    ]

    return {
        "valid_bag": bool(valid_bags),
        "has_defects": bool(valid_defects),
        "defects": list(set(d.defect_type for d in valid_defects)),
        "main_confidence": max((d.confidence for d in detections), default=0),
    }
