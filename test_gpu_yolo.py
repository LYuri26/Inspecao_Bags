import torch
from ultralytics import YOLO
import pynvml


def check_torch():
    print("=" * 50)
    print("🔎 Verificando PyTorch...")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Versão CUDA (PyTorch): {torch.version.cuda}")
        print(f"Dispositivo atual: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Multiprocessadores: {props.multi_processor_count}")
        print(f"Memória total: {round(props.total_memory / 1024**3, 2)} GB")


def check_nvidia():
    print("=" * 50)
    print("🔎 Verificando NVIDIA NVML...")
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        # Decodifica bytes somente se necessário
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU detectada: {name}")
        print(f"Memória total: {round(mem_info.total / 1024**3, 2)} GB")
        print(f"Memória usada: {round(mem_info.used / 1024**3, 2)} GB")
        print(f"Memória livre: {round(mem_info.free / 1024**3, 2)} GB")
    except Exception as e:
        print("Erro ao acessar NVML:", e)


def test_yolo():
    print("=" * 50)
    print("🚀 Testando YOLOv8...")
    try:
        # Permite explicitamente a classe DetectionModel (PyTorch 2.6+)
        from ultralytics.nn.tasks import DetectionModel

        torch.serialization.add_safe_globals([DetectionModel])

        model = YOLO("yolov8n.pt")  # modelo pequeno
        results = model.predict(
            source="https://ultralytics.com/images/bus.jpg",
            device=0,  # força GPU
            show=False,
        )
        print("✅ YOLOv8 rodou com sucesso na GPU!")
        print("Classes detectadas:", results[0].names)
        print("Objetos detectados:", results[0].boxes.shape[0])
    except Exception as e:
        print("❌ Erro ao rodar YOLOv8:", e)


if __name__ == "__main__":
    check_torch()
    check_nvidia()
    test_yolo()
    print("=" * 50)
