#!/usr/bin/env python3
"""
Script de diagnóstico completo para GPU, PyTorch, CUDA e YOLOv8
Testa todas as versões do YOLOv8 (s, m, l, x) e verifica desempenho
"""

import os
import sys
import platform
import subprocess
import time
import logging
from pathlib import Path
import torch
import cv2
import numpy as np
from ultralytics import YOLO

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("gpu_diagnostic.log")],
)
logger = logging.getLogger(__name__)


class GPUDiagnostic:
    def __init__(self):
        self.results = {}
        self.test_image = self.create_test_image()

    def create_test_image(self):
        """Cria uma imagem de teste para inferência"""
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        cv2.imwrite("test_image.jpg", img)
        return "test_image.jpg"

    def run_system_command(self, cmd):
        """Executa comando do sistema e retorna resultado"""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Timeout", 1
        except Exception as e:
            return "", str(e), 1

    def check_system_info(self):
        """Verifica informações do sistema"""
        logger.info("=== INFORMAÇÕES DO SISTEMA ===")

        info = {
            "Sistema": platform.system(),
            "Versão": platform.version(),
            "Arquitetura": platform.architecture(),
            "Processador": platform.processor(),
            "Python": platform.python_version(),
            "CPUs": os.cpu_count(),
        }

        for key, value in info.items():
            logger.info(f"{key}: {value}")
            self.results[f"system_{key.lower()}"] = value

        return info

    def check_nvidia_drivers(self):
        """Verifica drivers NVIDIA"""
        logger.info("\n=== DRIVERS NVIDIA ===")

        driver_info = {}

        if platform.system() == "Windows":
            # Verifica drivers no Windows
            try:
                import winreg

                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SYSTEM\CurrentControlSet\Services\nvlddmkm",
                )
                driver_version = winreg.QueryValueEx(key, "Version")[0]
                winreg.CloseKey(key)
                driver_info["version"] = driver_version
                logger.info(f"Driver NVIDIA: {driver_version}")
            except:
                logger.warning("Driver NVIDIA não encontrado no registro")
        else:
            # Linux/Mac
            stdout, stderr, code = self.run_system_command("nvidia-smi")
            if code == 0:
                driver_info["nvidia_smi"] = stdout
                logger.info("nvidia-smi disponível")
                # Extrai versão do driver
                for line in stdout.split("\n"):
                    if "Driver Version" in line:
                        driver_info["version"] = line.strip()
                        logger.info(line.strip())
            else:
                logger.warning("nvidia-smi não disponível")

        self.results["nvidia_drivers"] = driver_info
        return driver_info

    def check_cuda_installation(self):
        """Verifica instalação CUDA"""
        logger.info("\n=== INSTALAÇÃO CUDA ===")

        cuda_info = {}

        # Verifica nvcc
        stdout, stderr, code = self.run_system_command("nvcc --version")
        if code == 0:
            cuda_info["nvcc"] = stdout.split("\n")[0]
            first_line = stdout.splitlines()[0] if stdout else "N/A"
            logger.info(f"NVCC: {first_line}")
        else:
            logger.warning("NVCC não encontrado")

        # Verifica variáveis de ambiente CUDA
        cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
        if cuda_path:
            cuda_info["cuda_path"] = cuda_path
            logger.info(f"CUDA_PATH: {cuda_path}")

            # Verifica se binários CUDA existem
            cuda_bin = Path(cuda_path) / "bin"
            if cuda_bin.exists():
                cuda_info["cuda_bin_exists"] = True
                logger.info("✅ Binários CUDA encontrados")
            else:
                cuda_info["cuda_bin_exists"] = False
                logger.warning("❌ Binários CUDA não encontrados")
        else:
            logger.warning("Variáveis de ambiente CUDA não configuradas")

        self.results["cuda_installation"] = cuda_info
        return cuda_info

    def check_pytorch_gpu(self):
        """Verifica PyTorch e GPU"""
        logger.info("\n=== PyTorch & GPU ===")

        torch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "cudnn_version": (
                torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A"
            ),
        }

        logger.info(f"PyTorch Version: {torch_info['version']}")
        logger.info(f"CUDA Available: {torch_info['cuda_available']}")

        if torch_info["cuda_available"]:
            logger.info(f"CUDA Version: {torch_info['cuda_version']}")
            logger.info(f"cuDNN Version: {torch_info['cudnn_version']}")

            # Informações das GPUs
            device_count = torch.cuda.device_count()
            torch_info["device_count"] = device_count
            logger.info(f"Number of GPUs: {device_count}")

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                torch_info[f"gpu_{i}_name"] = props.name
                torch_info[f"gpu_{i}_memory"] = f"{props.total_memory / 1024**3:.1f} GB"
                torch_info[f"gpu_{i}_capability"] = f"{props.major}.{props.minor}"

                logger.info(f"GPU {i}: {props.name}")
                logger.info(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
                logger.info(f"  Compute Capability: {props.major}.{props.minor}")
        else:
            logger.warning("❌ CUDA não disponível no PyTorch")

        self.results["pytorch_gpu"] = torch_info
        return torch_info

    def test_gpu_performance(self):
        """Testa performance da GPU com operações simples"""
        logger.info("\n=== TESTE DE PERFORMANCE GPU ===")

        if not torch.cuda.is_available():
            logger.warning("GPU não disponível para teste de performance")
            return {}

        performance = {}

        # Teste de transferência CPU->GPU
        start_time = time.time()
        x = torch.randn(10000, 10000)
        x_gpu = x.cuda()
        transfer_time = time.time() - start_time
        performance["transfer_cpu_to_gpu"] = transfer_time
        logger.info(f"Transferência CPU->GPU: {transfer_time:.4f}s")

        # Teste de operação matricial
        start_time = time.time()
        result = torch.mm(x_gpu, x_gpu.t())
        op_time = time.time() - start_time
        performance["matrix_operation"] = op_time
        logger.info(f"Operação matricial: {op_time:.4f}s")

        # Teste de transferência GPU->CPU
        start_time = time.time()
        result_cpu = result.cpu()
        transfer_back_time = time.time() - start_time
        performance["transfer_gpu_to_cpu"] = transfer_back_time
        logger.info(f"Transferência GPU->CPU: {transfer_back_time:.4f}s")

        self.results["gpu_performance"] = performance
        return performance

    def test_yolov8_model(self, model_size="s"):
        """Testa modelo YOLOv8 específico"""
        logger.info(f"\n=== TESTE YOLOv8-{model_size.upper()} ===")

        model_info = {}

        try:
            # Carrega modelo
            start_time = time.time()
            model = YOLO(f"yolov8{model_size}.pt")
            load_time = time.time() - start_time
            model_info["load_time"] = load_time
            logger.info(f"Modelo carregado em: {load_time:.2f}s")

            # Warmup
            _ = model(self.test_image, verbose=False)

            # Teste de inferência
            start_time = time.time()
            results = model(self.test_image, verbose=False)
            inference_time = time.time() - start_time
            model_info["inference_time"] = inference_time
            logger.info(f"Inferência: {inference_time:.3f}s")

            # Verifica se usou GPU
            device = next(model.model.parameters()).device
            model_info["device"] = str(device)
            logger.info(f"Dispositivo usado: {device}")

            # Informações do modelo
            model_info["parameters"] = sum(p.numel() for p in model.model.parameters())
            logger.info(f"Parâmetros: {model_info['parameters']:,}")

            if len(results) > 0:
                model_info["detections"] = (
                    len(results[0].boxes) if results[0].boxes else 0
                )
                logger.info(f"Detecções: {model_info['detections']}")

        except Exception as e:
            model_info["error"] = str(e)
            logger.error(f"Erro no teste YOLOv8-{model_size}: {e}")

        self.results[f"yolov8_{model_size}"] = model_info
        return model_info

    def test_all_yolov8_models(self):
        """Testa todas as versões do YOLOv8"""
        logger.info("\n" + "=" * 50)
        logger.info("TESTE COMPLETO YOLOv8")
        logger.info("=" * 50)

        models = ["s", "m", "l", "x"]
        results = {}

        for model_size in models:
            results[model_size] = self.test_yolov8_model(model_size)
            time.sleep(2)  # Pausa entre testes

        return results

    def generate_report(self):
        """Gera relatório completo"""
        logger.info("\n" + "=" * 60)
        logger.info("RELATÓRIO COMPLETO DE DIAGNÓSTICO")
        logger.info("=" * 60)

        # Resumo geral
        cuda_available = self.results.get("pytorch_gpu", {}).get(
            "cuda_available", False
        )
        logger.info(f"✅ CUDA Disponível: {cuda_available}")

        if cuda_available:
            gpu_name = self.results["pytorch_gpu"].get("gpu_0_name", "N/A")
            gpu_memory = self.results["pytorch_gpu"].get("gpu_0_memory", "N/A")
            logger.info(f"✅ GPU: {gpu_name} ({gpu_memory})")

            # Performance YOLOv8
            logger.info("\n📊 Performance YOLOv8:")
            for model_size in ["s", "m", "l", "x"]:
                model_key = f"yolov8_{model_size}"
                if model_key in self.results:
                    info = self.results[model_key]
                    if "inference_time" in info:
                        logger.info(
                            f"  YOLOv8-{model_size.upper()}: {info['inference_time']:.3f}s"
                        )

        # Recomendações
        logger.info("\n💡 RECOMENDAÇÕES:")
        if not cuda_available:
            logger.info("❌ Instale PyTorch com suporte CUDA")
            logger.info("👉 pip uninstall torch torchvision torchaudio")
            logger.info(
                "👉 pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121"
            )
        else:
            logger.info("✅ Sistema configurado corretamente para YOLOv8")
            logger.info("✅ Recomendado usar YOLOv8-s para maior velocidade")
            logger.info("✅ Use YOLOv8-m ou YOLOv8-l para melhor precisão")

        return self.results

    def run_full_diagnostic(self):
        """Executa diagnóstico completo"""
        try:
            self.check_system_info()
            self.check_nvidia_drivers()
            self.check_cuda_installation()
            self.check_pytorch_gpu()

            if torch.cuda.is_available():
                self.test_gpu_performance()
                self.test_all_yolov8_models()
            else:
                logger.warning("Pulando testes de GPU - CUDA não disponível")

            return self.generate_report()

        except Exception as e:
            logger.error(f"Erro durante diagnóstico: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return {}


def main():
    """Função principal"""
    print("🔍 Iniciando diagnóstico completo de GPU e YOLOv8...")
    print("⏰ Isso pode levar alguns minutos...\n")

    diagnostic = GPUDiagnostic()
    results = diagnostic.run_full_diagnostic()

    # Salva resultados em arquivo
    import json

    with open("diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Diagnóstico completo! Resultados salvos em 'diagnostic_results.json'")
    print("📋 Verifique o arquivo 'gpu_diagnostic.log' para detalhes completos")

    return 0


if __name__ == "__main__":
    sys.exit(main())
