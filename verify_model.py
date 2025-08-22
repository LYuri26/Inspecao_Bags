"""
Script de verificação de desempenho do detector de sacolas.
Compara predições do modelo com ground truth e exibe resultados visuais.
"""

import os
import random
import logging
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from src.detector.detector import BagDetector

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("verification.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Classes do dataset (deve corresponder ao config.yaml do treinamento)
CLASSES = ["Sacola", "Rasgo", "Corte", "Mancha", "Descostura", "Sujeira"]

# Cores para visualização (BGR)
COLORS = {
    "gt": (255, 0, 0),  # Azul para ground truth
    "pred": (0, 255, 0),  # Verde para predições
    "fp": (0, 0, 255),  # Vermelho para falsos positivos
    "fn": (255, 255, 0),  # Ciano para falsos negativos
}


def setup_directories() -> Tuple[Path, Path]:
    """Configura e valida os diretórios do dataset."""
    dataset_dir = Path("dataset_sacolas/images/test")
    labels_dir = Path("dataset_sacolas/labels/test")

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {dataset_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Diretório de labels não encontrado: {labels_dir}")

    return dataset_dir, labels_dir


def load_labels(
    label_path: Path, img_shape: Tuple[int, int, int]
) -> List[Dict[str, Any]]:
    """Carrega labels YOLO (txt) e converte para coordenadas absolutas."""
    h, w = img_shape[:2]
    boxes = []

    if not label_path.exists():
        return boxes

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f.readlines(), 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    cls, x, y, bw, bh = map(float, line.split())
                    cls = int(cls)

                    # Validação dos valores
                    if not (0 <= cls < len(CLASSES)):
                        logger.warning(
                            f"Classe inválida {cls} em {label_path.name}:{line_num}"
                        )
                        continue

                    # Conversão YOLO -> coordenadas absolutas
                    x1 = max(0, int((x - bw / 2) * w))
                    y1 = max(0, int((y - bh / 2) * h))
                    x2 = min(w, int((x + bw / 2) * w))
                    y2 = min(h, int((y + bh / 2) * h))

                    # Validação da bounding box
                    if x1 >= x2 or y1 >= y2:
                        logger.warning(
                            f"Bounding box inválida em {label_path.name}:{line_num}"
                        )
                        continue

                    boxes.append(
                        {
                            "class_id": cls,
                            "class_name": CLASSES[cls],
                            "bbox": [x1, y1, x2, y2],
                        }
                    )

                except ValueError as e:
                    logger.error(
                        f"Erro ao processar linha {line_num} em {label_path.name}: {e}"
                    )
                    continue

    except IOError as e:
        logger.error(f"Erro ao ler arquivo {label_path}: {e}")

    return boxes


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calcula Intersection over Union entre duas bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Coordenadas da interseção
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    # Área da interseção
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Área das boxes individuais
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Área da união
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def draw_boxes(
    img: np.ndarray,
    boxes: List[Dict[str, Any]],
    color: Tuple[int, int, int],
    prefix: str = "",
    show_conf: bool = False,
) -> np.ndarray:
    """Desenha caixas e labels na imagem."""
    img_copy = img.copy()

    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]

        # Label com ou sem confiança
        if show_conf and "confidence" in box:
            label = f"{prefix}{box['class_name']} {box['confidence']:.2f}"
        else:
            label = f"{prefix}{box['class_name']}"

        # Desenha retângulo
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

        # Background do texto para melhor legibilidade
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        cv2.rectangle(
            img_copy, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1
        )

        # Texto
        cv2.putText(
            img_copy,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # Texto branco
            2,
        )

    return img_copy


def evaluate_detections(
    gt_boxes: List[Dict[str, Any]],
    pred_boxes: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Avalia as detecções comparando com ground truth."""
    metrics = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "matched_pairs": [],
        "false_positives_list": [],
        "false_negatives_list": [],
    }

    # Para cada ground truth, encontra a melhor predição correspondente
    gt_matched = [False] * len(gt_boxes)
    pred_matched = [False] * len(pred_boxes)

    for i, gt_box in enumerate(gt_boxes):
        best_iou = 0
        best_j = -1

        for j, pred_box in enumerate(pred_boxes):
            if pred_matched[j]:
                continue

            if gt_box["class_id"] == pred_box["class_id"]:
                iou = calculate_iou(gt_box["bbox"], pred_box["bbox"])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_j = j

        if best_j != -1:
            metrics["true_positives"] += 1
            gt_matched[i] = True
            pred_matched[best_j] = True
            metrics["matched_pairs"].append((gt_box, pred_boxes[best_j], best_iou))

    # Falsos positivos (predições não correspondidas)
    for j, matched in enumerate(pred_matched):
        if not matched:
            metrics["false_positives"] += 1
            metrics["false_positives_list"].append(pred_boxes[j])

    # Falsos negativos (ground truths não correspondidos)
    for i, matched in enumerate(gt_matched):
        if not matched:
            metrics["false_negatives"] += 1
            metrics["false_negatives_list"].append(gt_boxes[i])

    # Cálculo de métricas
    precision = (
        metrics["true_positives"]
        / (metrics["true_positives"] + metrics["false_positives"])
        if (metrics["true_positives"] + metrics["false_positives"]) > 0
        else 0
    )
    recall = (
        metrics["true_positives"]
        / (metrics["true_positives"] + metrics["false_negatives"])
        if (metrics["true_positives"] + metrics["false_negatives"]) > 0
        else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    metrics.update({"precision": precision, "recall": recall, "f1_score": f1_score})

    return metrics


def process_image(
    detector: BagDetector,
    img_path: Path,
    labels_dir: Path,
    display_time: int = 3,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Processa uma imagem individual e retorna métricas."""
    # Carrega imagem
    img = cv2.imread(str(img_path))
    if img is None:
        logger.error(f"❌ Erro ao carregar imagem: {img_path}")
        return {}

    # Ground truth
    label_path = labels_dir / (img_path.stem + ".txt")
    gt_boxes = load_labels(label_path, img.shape)

    logger.info(f"\n🔍 Processando: {img_path.name}")
    logger.info(f"🎯 Ground truths: {len(gt_boxes)}")

    # Predições
    start_time = time.time()
    pred_boxes = detector.detect(img)
    inference_time = (time.time() - start_time) * 1000  # ms

    logger.info(f"✅ Predições: {len(pred_boxes)}")
    logger.info(f"⏱️ Tempo de inferência: {inference_time:.1f}ms")

    # Avaliação
    metrics = evaluate_detections(gt_boxes, pred_boxes, iou_threshold)

    logger.info(f"📊 True Positives: {metrics['true_positives']}")
    logger.info(f"📊 False Positives: {metrics['false_positives']}")
    logger.info(f"📊 False Negatives: {metrics['false_negatives']}")
    logger.info(f"📊 Precision: {metrics['precision']:.3f}")
    logger.info(f"📊 Recall: {metrics['recall']:.3f}")
    logger.info(f"📊 F1-Score: {metrics['f1_score']:.3f}")

    # Visualização
    vis_img = img.copy()

    # Ground truth (azul)
    vis_img = draw_boxes(vis_img, gt_boxes, COLORS["gt"], "GT: ")

    # Predições corretas (verde)
    vis_img = draw_boxes(
        vis_img,
        [p for _, p, _ in metrics["matched_pairs"]],
        COLORS["pred"],
        "TP: ",
        True,
    )

    # Falsos positivos (vermelho)
    vis_img = draw_boxes(
        vis_img, metrics["false_positives_list"], COLORS["fp"], "FP: ", True
    )

    # Falsos negativos (ciano)
    vis_img = draw_boxes(vis_img, metrics["false_negatives_list"], COLORS["fn"], "FN: ")

    # Adiciona informações da imagem
    cv2.putText(
        vis_img,
        f"Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | F1: {metrics['f1_score']:.3f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        vis_img,
        f"Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | F1: {metrics['f1_score']:.3f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Exibe resultado
    cv2.imshow("Verificação - Pressione qualquer tecla para continuar", vis_img)
    cv2.waitKey(display_time * 1000)

    return metrics


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Verificação do detector de sacolas")
    parser.add_argument(
        "--num_images", type=int, default=6, help="Número de imagens para testar"
    )
    parser.add_argument(
        "--display_time",
        type=int,
        default=3,
        help="Tempo de exibição por imagem (segundos)",
    )
    parser.add_argument(
        "--iou_threshold", type=float, default=0.5, help="Threshold IoU para matching"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="runs/train/detector_sacola/weights/best.pt",
        help="Caminho do modelo",
    )
    args = parser.parse_args()

    try:
        logger.info("🚀 Iniciando verificação do detector...")

        # Carrega detector
        detector = BagDetector(model_path=args.model_path, confidence_threshold=0.2)
        logger.info("✅ Detector carregado com sucesso")

        # Configura diretórios
        dataset_dir, labels_dir = setup_directories()

        # Lista imagens
        images = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.png"))
        if not images:
            logger.error("❌ Nenhuma imagem encontrada no dataset de teste.")
            return

        # Seleciona imagens aleatórias
        selected_images = random.sample(images, min(args.num_images, len(images)))
        logger.info(
            f"\n🖼️ Imagens selecionadas para verificação ({len(selected_images)}):"
        )
        for img in selected_images:
            logger.info(f"- {img.name}")

        # Processa imagens
        all_metrics = []
        for img_path in selected_images:
            metrics = process_image(
                detector, img_path, labels_dir, args.display_time, args.iou_threshold
            )
            if metrics:
                all_metrics.append(metrics)

        # Calcula métricas agregadas
        if all_metrics:
            avg_precision = np.mean([m["precision"] for m in all_metrics])
            avg_recall = np.mean([m["recall"] for m in all_metrics])
            avg_f1 = np.mean([m["f1_score"] for m in all_metrics])

            logger.info(f"\n📈 MÉTRICAS AGREGADAS:")
            logger.info(f"📊 Precision média: {avg_precision:.3f}")
            logger.info(f"📊 Recall médio: {avg_recall:.3f}")
            logger.info(f"📊 F1-Score médio: {avg_f1:.3f}")

        logger.info("\n✅ Verificação concluída com sucesso")

    except Exception as e:
        logger.error(f"❌ Erro durante a verificação: {e}")
        raise


if __name__ == "__main__":
    main()
