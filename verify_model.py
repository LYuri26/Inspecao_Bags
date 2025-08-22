import sys
import os
import cv2
from pathlib import Path
from random import sample
import time

# Configuração de paths
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from src.detector.detector import BagDetector


def setup_environment():
    """Configura variáveis de ambiente para evitar warnings do Qt"""
    os.environ["QT_QPA_PLATFORM"] = "xcb"  # Alternativa para sistemas Linux


def select_test_images(dataset_path):
    """Seleciona imagens para teste de forma balanceada"""
    train_images = list((dataset_path / "train").glob("*.jpg"))
    val_images = list((dataset_path / "val").glob("*.jpg"))

    # Seleciona 2 de cada (ou menos se não houver suficientes)
    num_samples = min(2, len(train_images)), min(2, len(val_images))
    return [*sample(train_images, num_samples[0]), *sample(val_images, num_samples[1])]


def process_image(detector, img_path, display_time=5):
    """Processa uma imagem e mostra os resultados"""
    print(f"\n🔍 Processando: {img_path.name}")

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Erro ao carregar a imagem: {img_path}")
        return

    # Detecção
    start_time = time.time()
    detections = detector.detect(img)
    inference_time = (time.time() - start_time) * 1000  # ms

    # Resultados
    if not detections:
        print("Nenhum objeto detectado")
    else:
        for d in detections:
            print(f"- {d['class_name']} (confiança: {d['confidence']:.2f})")
    print(f"⏱️ Tempo total: {inference_time:.1f}ms")

    # Visualização
    result_img = detector.draw_detections(img, detections)
    cv2.imshow("Resultado - Pressione qualquer tecla para continuar", result_img)
    cv2.waitKey(display_time * 1000)  # Mostra por N segundos ou até tecla


def main():
    setup_environment()

    # Configuração de paths
    model_path = project_root / "modelos" / "detector_sacola.pt"
    dataset_path = project_root / "dataset_sacolas" / "images"

    # Verificações
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}")

    # Carregar detector
    detector = BagDetector(str(model_path))
    print("\n✅ Detector carregado com sucesso")
    print(f"📂 Dataset: {dataset_path}")

    # Selecionar imagens
    test_images = select_test_images(dataset_path)
    print(f"\n🖼️ Imagens selecionadas para teste:")
    for img in test_images:
        print(f"- {img.name}")

    # Processar imagens
    for img_path in test_images:
        process_image(detector, img_path, display_time=5)

    cv2.destroyAllWindows()
    print("\n✅ Teste concluído com sucesso")


if __name__ == "__main__":
    main()
