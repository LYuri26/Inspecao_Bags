#!/usr/bin/env python3
"""
DIAGNÓSTICO COMPLETO - GPU, PyTorch, CUDA, YOLOv8
Script combinado para análise completa do sistema e desempenho
"""

import os
import sys
import platform
import subprocess
import time
import logging
import json
import psutil
import pkg_resources
from pathlib import Path
from datetime import datetime
import torch
import cv2
import numpy as np
from ultralytics import YOLO

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("diagnostico_completo.log")],
)
logger = logging.getLogger(__name__)


class DiagnosticCompleto:
    def __init__(self):
        self.results = {}
        self.report_lines = []
        self.test_image = self.create_test_image()

    def create_test_image(self):
        """Cria uma imagem de teste para inferência"""
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        cv2.imwrite("test_image.jpg", img)
        return "test_image.jpg"

    def add_to_report(self, section, content):
        """Adiciona conteúdo ao relatório"""
        if isinstance(content, str):
            self.report_lines.append(content)
        elif isinstance(content, list):
            self.report_lines.extend(content)
        else:
            self.report_lines.append(str(content))

    def run_cmd(self, cmd):
        """Executa comando no terminal e retorna saída"""
        try:
            result = subprocess.check_output(
                cmd, shell=True, stderr=subprocess.STDOUT, text=True, timeout=30
            )
            return result.strip()
        except subprocess.CalledProcessError as e:
            return f"Erro ao executar '{cmd}':\n{e.output}"
        except FileNotFoundError:
            return f"Comando '{cmd}' não encontrado"
        except subprocess.TimeoutExpired:
            return "Timeout ao executar comando"

    def get_installed_packages(self):
        """Lista pacotes relacionados a PyTorch e CUDA"""
        packages = []
        for dist in pkg_resources.working_set:
            if (
                "torch" in dist.key.lower()
                or "cuda" in dist.key.lower()
                or "nvidia" in dist.key.lower()
                or "ultralytics" in dist.key.lower()
                or "opencv" in dist.key.lower()
                or "albumentation" in dist.key.lower()
            ):
                packages.append(f"{dist.project_name}=={dist.version}")
        return sorted(packages)

    def check_system_info(self):
        """Verifica informações do sistema"""
        logger.info("=== INFORMAÇÕES DO SISTEMA ===")

        system_info = {
            "Sistema": platform.system(),
            "Versão": platform.version(),
            "Arquitetura": platform.architecture()[0],
            "Processador": platform.processor(),
            "Python": platform.python_version(),
            "CPUs Físicos": psutil.cpu_count(logical=False),
            "CPUs Lógicos": psutil.cpu_count(logical=True),
            "RAM Total": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.add_to_report("SISTEMA", ["=== SISTEMA OPERACIONAL ==="])
        for key, value in system_info.items():
            line = f"{key}: {value}"
            logger.info(line)
            self.add_to_report("SISTEMA", line)

        # Distribuição Linux
        if platform.system() == "Linux":
            distro = self.run_cmd("lsb_release -d")
            self.add_to_report("SISTEMA", f"Distribuição: {distro}")

        self.results["system_info"] = system_info
        return system_info

    def check_nvidia_drivers(self):
        """Verifica drivers NVIDIA"""
        logger.info("\n=== DRIVERS NVIDIA ===")
        self.add_to_report("GPU", ["\n=== GPU (via nvidia-smi) ==="])

        driver_info = {}
        nvidia_smi = self.run_cmd("nvidia-smi")

        self.add_to_report("GPU", nvidia_smi)

        if platform.system() == "Windows":
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
                self.add_to_report("GPU", f"Driver NVIDIA: {driver_version}")
            except Exception as e:
                logger.warning("Driver NVIDIA não encontrado no registro")
                self.add_to_report("GPU", "Driver NVIDIA não encontrado no registro")
        else:
            if "NVIDIA-SMI" in nvidia_smi:
                driver_info["nvidia_smi"] = True
                for line in nvidia_smi.split("\n"):
                    if "Driver Version" in line:
                        driver_info["version"] = line.strip()
                        logger.info(line.strip())
                        self.add_to_report("GPU", line.strip())
            else:
                logger.warning("nvidia-smi não disponível")
                self.add_to_report("GPU", "nvidia-smi não disponível")

        self.results["nvidia_drivers"] = driver_info
        return driver_info

    def check_cuda_installation(self):
        """Verifica instalação CUDA"""
        logger.info("\n=== INSTALAÇÃO CUDA ===")
        self.add_to_report("CUDA", ["\n=== CUDA (sistema) ==="])

        cuda_info = {}

        # Verifica nvcc
        nvcc_output = self.run_cmd("nvcc --version")
        self.add_to_report("CUDA", nvcc_output)

        if "release" in nvcc_output.lower():
            cuda_info["nvcc"] = nvcc_output.split("\n")[0]
            logger.info(f"NVCC: {nvcc_output.splitlines()[0]}")
        else:
            logger.warning("NVCC não encontrado")
            self.add_to_report("CUDA", "NVCC não encontrado")

        # Verifica variáveis de ambiente CUDA
        cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
        if cuda_path:
            cuda_info["cuda_path"] = cuda_path
            logger.info(f"CUDA_PATH: {cuda_path}")
            self.add_to_report("CUDA", f"CUDA_PATH: {cuda_path}")

            # Verifica se binários CUDA existem
            cuda_bin = Path(cuda_path) / "bin"
            if cuda_bin.exists():
                cuda_info["cuda_bin_exists"] = True
                logger.info("✅ Binários CUDA encontrados")
                self.add_to_report("CUDA", "✅ Binários CUDA encontrados")
            else:
                cuda_info["cuda_bin_exists"] = False
                logger.warning("❌ Binários CUDA não encontrados")
                self.add_to_report("CUDA", "❌ Binários CUDA não encontrados")
        else:
            logger.warning("Variáveis de ambiente CUDA não configuradas")
            self.add_to_report("CUDA", "Variáveis de ambiente CUDA não configuradas")

        self.results["cuda_installation"] = cuda_info
        return cuda_info

    def check_pytorch_gpu(self):
        """Verifica PyTorch e GPU"""
        logger.info("\n=== PyTorch & GPU ===")
        self.add_to_report("PyTorch", ["\n=== PyTorch ==="])

        torch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "cudnn_version": (
                torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A"
            ),
        }

        # Adiciona informações ao relatório
        self.add_to_report("PyTorch", f"torch.__version__: {torch_info['version']}")
        self.add_to_report(
            "PyTorch", f"CUDA disponível: {torch_info['cuda_available']}"
        )

        logger.info(f"PyTorch Version: {torch_info['version']}")
        logger.info(f"CUDA Available: {torch_info['cuda_available']}")

        if torch_info["cuda_available"]:
            self.add_to_report(
                "PyTorch", f"Versão CUDA (PyTorch): {torch_info['cuda_version']}"
            )
            self.add_to_report(
                "PyTorch", f"Versão cuDNN: {torch_info['cudnn_version']}"
            )

            logger.info(f"CUDA Version: {torch_info['cuda_version']}")
            logger.info(f"cuDNN Version: {torch_info['cudnn_version']}")

            # Informações das GPUs
            device_count = torch.cuda.device_count()
            torch_info["device_count"] = device_count
            self.add_to_report("PyTorch", f"Número de GPUs: {device_count}")
            logger.info(f"Number of GPUs: {device_count}")

            self.add_to_report("GPU", ["\n=== GPU (via torch) ==="])
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                torch_info[f"gpu_{i}_name"] = props.name
                torch_info[f"gpu_{i}_memory"] = f"{props.total_memory / 1024**3:.1f} GB"
                torch_info[f"gpu_{i}_capability"] = f"{props.major}.{props.minor}"

                gpu_line = f"GPU {i}: {props.name} | Memória: {props.total_memory / 1024**3:.1f} GB | Multiprocessadores: {props.multi_processor_count}"
                self.add_to_report("GPU", gpu_line)
                logger.info(f"GPU {i}: {props.name}")
                logger.info(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
                logger.info(f"  Compute Capability: {props.major}.{props.minor}")
        else:
            self.add_to_report("PyTorch", "Nenhuma GPU CUDA detectada pelo PyTorch")
            logger.warning("❌ CUDA não disponível no PyTorch")

        self.results["pytorch_gpu"] = torch_info
        return torch_info

    def check_installed_packages(self):
        """Verifica pacotes instalados"""
        logger.info("\n=== PACOTES INSTALADOS ===")
        self.add_to_report("Pacotes", ["\n=== Pacotes relevantes instalados ==="])

        packages = self.get_installed_packages()
        if packages:
            for pkg in packages:
                self.add_to_report("Pacotes", pkg)
                logger.info(pkg)
        else:
            self.add_to_report("Pacotes", "Nenhum pacote relevante encontrado")
            logger.info("Nenhum pacote relevante encontrado")

        self.results["packages"] = packages
        return packages

    def test_gpu_performance(self):
        """Testa performance da GPU com operações simples"""
        logger.info("\n=== TESTE DE PERFORMANCE GPU ===")
        self.add_to_report("Performance", ["\n=== TESTE DE PERFORMANCE GPU ==="])

        if not torch.cuda.is_available():
            self.add_to_report(
                "Performance", "GPU não disponível para teste de performance"
            )
            logger.warning("GPU não disponível para teste de performance")
            return {}

        performance = {}

        # Teste de transferência CPU->GPU
        start_time = time.time()
        x = torch.randn(5000, 5000)  # Matriz menor para evitar memory error
        x_gpu = x.cuda()
        transfer_time = time.time() - start_time
        performance["transfer_cpu_to_gpu"] = transfer_time
        line = f"Transferência CPU->GPU: {transfer_time:.4f}s"
        self.add_to_report("Performance", line)
        logger.info(line)

        # Teste de operação matricial
        start_time = time.time()
        result = torch.mm(x_gpu, x_gpu.t())
        op_time = time.time() - start_time
        performance["matrix_operation"] = op_time
        line = f"Operação matricial: {op_time:.4f}s"
        self.add_to_report("Performance", line)
        logger.info(line)

        # Teste de transferência GPU->CPU
        start_time = time.time()
        result_cpu = result.cpu()
        transfer_back_time = time.time() - start_time
        performance["transfer_gpu_to_cpu"] = transfer_back_time
        line = f"Transferência GPU->CPU: {transfer_back_time:.4f}s"
        self.add_to_report("Performance", line)
        logger.info(line)

        # Limpeza
        del x, x_gpu, result, result_cpu
        torch.cuda.empty_cache()

        self.results["gpu_performance"] = performance
        return performance

    def test_yolov8_model(self, model_size="s"):
        """Testa modelo YOLOv8 específico"""
        logger.info(f"\n=== TESTE YOLOv8-{model_size.upper()} ===")
        self.add_to_report("YOLOv8", [f"\n=== TESTE YOLOv8-{model_size.upper()} ==="])

        model_info = {}

        try:
            # Carrega modelo
            start_time = time.time()
            model = YOLO(f"yolov8{model_size}.pt")
            load_time = time.time() - start_time
            model_info["load_time"] = load_time
            line = f"Modelo carregado em: {load_time:.2f}s"
            self.add_to_report("YOLOv8", line)
            logger.info(line)

            # Warmup
            _ = model(self.test_image, verbose=False)

            # Teste de inferência (múltiplas execuções para média)
            inference_times = []
            for _ in range(3):  # 3 execuções para média
                start_time = time.time()
                results = model(self.test_image, verbose=False)
                inference_times.append(time.time() - start_time)
                time.sleep(0.1)  # Pequena pausa

            inference_time = sum(inference_times) / len(inference_times)
            model_info["inference_time"] = inference_time
            line = f"Inferência (média): {inference_time:.3f}s"
            self.add_to_report("YOLOv8", line)
            logger.info(line)

            # Verifica se usou GPU
            device = next(model.model.parameters()).device
            model_info["device"] = str(device)
            line = f"Dispositivo usado: {device}"
            self.add_to_report("YOLOv8", line)
            logger.info(line)

            # Informações do modelo
            model_info["parameters"] = sum(p.numel() for p in model.model.parameters())
            line = f"Parâmetros: {model_info['parameters']:,}"
            self.add_to_report("YOLOv8", line)
            logger.info(line)

            if len(results) > 0 and hasattr(results[0], "boxes"):
                model_info["detections"] = (
                    len(results[0].boxes) if results[0].boxes else 0
                )
                line = f"Detecções: {model_info['detections']}"
                self.add_to_report("YOLOv8", line)
                logger.info(line)

            # Limpeza
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            model_info["error"] = str(e)
            line = f"Erro no teste YOLOv8-{model_size}: {e}"
            self.add_to_report("YOLOv8", line)
            logger.error(line)

        self.results[f"yolov8_{model_size}"] = model_info
        return model_info

    def test_all_yolov8_models(self):
        """Testa todas as versões do YOLOv8"""
        logger.info("\n" + "=" * 50)
        logger.info("TESTE COMPLETO YOLOv8")
        logger.info("=" * 50)
        self.add_to_report(
            "YOLOv8", ["\n" + "=" * 50, "TESTE COMPLETO YOLOv8", "=" * 50]
        )

        models = ["s", "m", "l", "x"]
        results = {}

        for model_size in models:
            results[model_size] = self.test_yolov8_model(model_size)
            time.sleep(1)  # Pausa entre testes

        return results

    def generate_recommendations(self):
        """Gera recomendações baseadas nos resultados"""
        logger.info("\n=== RECOMENDAÇÕES ===")
        self.add_to_report("Recomendações", ["\n=== RECOMENDAÇÕES ==="])

        recommendations = []

        # Verifica se PyTorch está usando CPU
        if "cpu" in torch.__version__.lower():
            recommendations.append("⚠️ Você está usando PyTorch apenas com CPU.")
            recommendations.append("👉 Para usar GPU NVIDIA, reinstale o PyTorch:")
            recommendations.append("   pip uninstall torch torchvision torchaudio -y")
            recommendations.append(
                "   pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121"
            )
        elif not torch.cuda.is_available():
            recommendations.append("⚠️ PyTorch não está detectando a GPU.")
            recommendations.append(
                "👉 Verifique se os drivers NVIDIA estão instalados corretamente"
            )
            recommendations.append("👉 Verifique se a GPU está funcionando no sistema")
        else:
            recommendations.append(
                "✅ PyTorch está configurado corretamente com suporte GPU"
            )

            # Recomendações de modelo YOLOv8 baseadas na performance
            if any(f"yolov8_{m}" in self.results for m in ["s", "m", "l", "x"]):
                recommendations.append("\n📊 Recomendações de modelo YOLOv8:")

                # Coleta tempos de inferência
                times = {}
                for model_size in ["s", "m", "l", "x"]:
                    key = f"yolov8_{model_size}"
                    if key in self.results and "inference_time" in self.results[key]:
                        times[model_size] = self.results[key]["inference_time"]

                if times:
                    # Ordena por velocidade
                    sorted_times = sorted(times.items(), key=lambda x: x[1])
                    fastest_model, fastest_time = sorted_times[0]
                    recommendations.append(
                        f"   🚀 Mais rápido: YOLOv8-{fastest_model.upper()} ({fastest_time:.3f}s)"
                    )

                    # Recomendação baseada na velocidade
                    if fastest_time < 0.1:
                        recommendations.append(
                            "   💡 Recomendado: YOLOv8-l ou YOLOv8-x para máxima precisão"
                        )
                    elif fastest_time < 0.2:
                        recommendations.append(
                            "   💡 Recomendado: YOLOv8-m para bom equilíbrio velocidade-precisão"
                        )
                    else:
                        recommendations.append(
                            "   💡 Recomendado: YOLOv8-s para máxima velocidade"
                        )

        # Adiciona recomendações ao relatório
        for rec in recommendations:
            self.add_to_report("Recomendações", rec)
            logger.info(rec)

        return recommendations

    def save_complete_report(self):
        """Salva relatório completo em arquivo TXT"""
        filename = "diagnostico_completo.txt"

        # Cabeçalho
        header = [
            "=" * 70,
            "DIAGNÓSTICO COMPLETO - GPU, PyTorch, CUDA, YOLOv8",
            f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
        ]

        # Conteúdo completo
        full_content = header + self.report_lines

        # Salva em arquivo
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(full_content))

        # Salva resultados JSON também
        with open("diagnostico_completo.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Relatório completo salvo em: {filename}")
        logger.info(f"✅ Dados detalhados salvos em: diagnostico_completo.json")

        return filename

    def run_full_diagnostic(self):
        """Executa diagnóstico completo"""
        try:
            logger.info("🔍 Iniciando diagnóstico completo...")
            print("🔍 Iniciando diagnóstico completo de GPU e YOLOv8...")
            print("⏰ Isso pode levar alguns minutos...\n")

            # Executa todas as verificações
            self.check_system_info()
            self.check_nvidia_drivers()
            self.check_cuda_installation()
            self.check_pytorch_gpu()
            self.check_installed_packages()

            # Testes de performance apenas se GPU disponível
            if torch.cuda.is_available():
                self.test_gpu_performance()
                self.test_all_yolov8_models()
            else:
                logger.warning("Pulando testes de performance - CUDA não disponível")
                self.add_to_report(
                    "Performance",
                    "⚠️ Testes de performance skipped - CUDA não disponível",
                )

            # Gera recomendações finais
            self.generate_recommendations()

            # Salva relatório completo
            report_file = self.save_complete_report()

            print(f"\n✅ Diagnóstico completo concluído!")
            print(f"📋 Relatório salvo em: {report_file}")
            print(f"📊 Dados detalhados: diagnostico_completo.json")
            print(f"📝 Logs completos: diagnostico_completo.log")

            return self.results

        except Exception as e:
            logger.error(f"Erro durante diagnóstico completo: {e}")
            import traceback

            logger.error(traceback.format_exc())

            # Tenta salvar relatório parcial
            try:
                self.save_complete_report()
            except:
                pass

            return {}


def main():
    """Função principal"""
    diagnostic = DiagnosticCompleto()
    results = diagnostic.run_full_diagnostic()
    return 0


if __name__ == "__main__":
    sys.exit(main())
