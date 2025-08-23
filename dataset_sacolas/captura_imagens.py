import cv2
import os
from datetime import datetime

# Pasta onde salvar as imagens
OUTPUT_DIR = "dataset/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tamanho para YOLOv8s
IMG_SIZE = 896


def letterbox_image(image, size=IMG_SIZE):
    """
    Redimensiona a imagem para quadrado (size x size) usando letterbox.
    Mantém proporção sem distorcer, preenchendo com preto.
    """
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)

    # Redimensiona mantendo proporção
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Cria fundo preto
    new_image = 255 * np.ones(
        (size, size, 3), dtype=np.uint8
    )  # fundo branco (pode trocar para 0 para preto)
    top = (size - nh) // 2
    left = (size - nw) // 2

    # Coloca a imagem no centro
    new_image[top : top + nh, left : left + nw] = resized
    return new_image


def capture_images(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame.")
            break

        # Redimensiona com letterbox
        frame_resized = letterbox_image(frame, IMG_SIZE)

        # Nome do arquivo
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)

        # Salva a imagem
        cv2.imwrite(filepath, frame_resized)
        print(f"✅ Imagem salva: {filepath}")

        count += 1

        # Mostra a imagem capturada
        cv2.imshow("Captura", frame_resized)

        # Captura a cada 90 segundos (1 min e meio)
        if cv2.waitKey(90000) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import numpy as np

    capture_images()
