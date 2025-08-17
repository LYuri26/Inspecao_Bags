import platform
import psutil
import subprocess
import sys

try:
    import torch
except ImportError:
    torch = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import ultralytics
except ImportError:
    ultralytics = None


def get_hardware_info():
    info = {}

    # Sistema
    info["Sistema"] = platform.system()
    info["Versão SO"] = platform.version()
    info["Release"] = platform.release()
    info["Arquitetura"] = platform.machine()

    # CPU
    info["Processador"] = platform.processor()
    info["Núcleos (físicos)"] = psutil.cpu_count(logical=False)
    info["Núcleos (lógicos)"] = psutil.cpu_count(logical=True)

    # Memória
    mem = psutil.virtual_memory()
    info["RAM Total (GB)"] = round(mem.total / (1024**3), 2)

    # GPU (via torch, se disponível)
    if torch and torch.cuda.is_available():
        info["GPU Disponível"] = torch.cuda.get_device_name(0)
        info["Versão CUDA"] = torch.version.cuda
    else:
        info["GPU Disponível"] = "Nenhuma (ou não detectada)"
        info["Versão CUDA"] = "N/A"

    return info


def get_yolo_compatibility():
    libs = {}

    # Torch
    if torch:
        libs["torch"] = f"Instalado (versão {torch.__version__})"
    else:
        libs["torch"] = "Não instalado"

    # OpenCV
    if cv2:
        libs["opencv-python"] = f"Instalado (versão {cv2.__version__})"
    else:
        libs["opencv-python"] = "Não instalado"

    # Ultralytics (YOLOv8)
    if ultralytics:
        libs["ultralytics"] = f"Instalado (versão {ultralytics.__version__})"
    else:
        libs["ultralytics"] = "Não instalado"

    return libs


if __name__ == "__main__":
    print("=" * 40)
    print("🔎 Informações de Hardware")
    print("=" * 40)
    for k, v in get_hardware_info().items():
        print(f"{k}: {v}")

    print("\n" + "=" * 40)
    print("📦 Bibliotecas para YOLO")
    print("=" * 40)
    for k, v in get_yolo_compatibility().items():
        print(f"{k}: {v}")

    print("\n✅ Script finalizado!")
