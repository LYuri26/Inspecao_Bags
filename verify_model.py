import os
import random
import logging
import time
import cv2
from pathlib import Path
from src.detector.detector import BagDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_image(detector, img_path, display_time=5):
    """Processa uma imagem, roda o modelo YOLO e exibe as detecções."""
    img = cv2.imread(str(img_path))
    if img is None:
        logger.error(f"❌ Erro ao carregar imagem: {img_path}")
        return

    logger.info(f"\n🔍 Processando: {os.path.basename(img_path)}")

    start_time = time.time()
    results = detector.model(img)  # roda YOLO diretamente

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append(
                {
                    "class_name": r.names[cls],
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                }
            )

    if detections:
        result_img = detector.draw_detections(img.copy(), detections)
        logger.info(f"✅ {len(detections)} objetos detectados")
    else:
        result_img = img.copy()
        logger.info("⚠️ Nenhum objeto detectado")

    end_time = time.time()
    logger.info(f"⏱️ Tempo total: {(end_time - start_time) * 1000:.1f}ms")

    # Exibe janela com resultado
    cv2.imshow("Resultado", result_img)
    cv2.waitKey(display_time * 1000)
    cv2.destroyAllWindows()


def main():
    logger.info("Carregando detector...")

    # Caminho correto do modelo
    model_path = "runs/train/detector_sacola/weights/best.pt"

    detector = BagDetector(model_path=model_path)
    logger.info("✅ Detector carregado com sucesso")

    dataset_dir = Path("dataset_sacolas/images/train")
    if not dataset_dir.exists():
        logger.error(f"❌ Diretório não encontrado: {dataset_dir}")
        return

    images = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.png"))
    if not images:
        logger.error("❌ Nenhuma imagem encontrada no dataset.")
        return

    selected_images = random.sample(images, min(4, len(images)))
    logger.info("\n🖼️ Imagens selecionadas para teste:")
    for img in selected_images:
        logger.info(f"- {img.name}")

    for img_path in selected_images:
        process_image(detector, img_path)

    logger.info("\n✅ Teste concluído com sucesso")


if __name__ == "__main__":
    main()
