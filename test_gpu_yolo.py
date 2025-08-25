import torch
from ultralytics import YOLO
import pynvml


# ---------------- Teste PyTorch + CUDA ----------------
def check_torch():
    print("=" * 50)
    print("🔎 Verificando PyTorch + CUDA")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Dispositivo atual: {torch.cuda.get_device_name(device)}")
        props = torch.cuda.get_device_properties(device)
        print(f"Multiprocessadores: {props.multi_processor_count}")
        print(f"Memória total: {round(props.total_memory / 1024**3, 2)} GB")
        # Teste rápido de tensor na GPU
        x = torch.randn((3, 3), device=device)
        y = torch.randn((3, 3), device=device)
        z = x @ y
        print("✅ Operação de teste na GPU executada com sucesso:", z)
    print("=" * 50)


# ---------------- Teste NVIDIA NVML ----------------
def check_nvidia():
    print("=" * 50)
    print("🔎 Verificando NVIDIA NVML...")
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU detectada: {name}")
        print(f"Memória total: {round(mem_info.total / 1024**3, 2)} GB")
        print(f"Memória usada: {round(mem_info.used / 1024**3, 2)} GB")
        print(f"Memória livre: {round(mem_info.free / 1024**3, 2)} GB")
    except Exception as e:
        print("Erro ao acessar NVML:", e)
    print("=" * 50)


# ---------------- Teste YOLOv8 ----------------
def test_yolo():
    print("=" * 50)
    print("🚀 Testando YOLOv8...")
    try:
        from ultralytics.nn.tasks import DetectionModel

        # Permite explicitamente a classe DetectionModel no torch.load
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
    print("=" * 50)


# ---------------- Main ----------------
if __name__ == "__main__":
    check_torch()
    check_nvidia()
    test_yolo()
