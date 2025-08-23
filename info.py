import subprocess
import sys
import platform
import pkg_resources
import torch
import psutil


def run_cmd(cmd):
    """Executa comando no terminal e retorna saída"""
    try:
        result = subprocess.check_output(
            cmd, shell=True, stderr=subprocess.STDOUT, text=True
        )
        return result.strip()
    except subprocess.CalledProcessError as e:
        return f"Erro ao executar '{cmd}':\n{e.output}"
    except FileNotFoundError:
        return f"Comando '{cmd}' não encontrado"


def get_installed_packages():
    """Lista pacotes relacionados a PyTorch e CUDA"""
    packages = []
    for dist in pkg_resources.working_set:
        if (
            "torch" in dist.key
            or "cuda" in dist.key
            or "nvidia" in dist.key
            or "ultralytics" in dist.key
            or "opencv" in dist.key
        ):
            packages.append(f"{dist.project_name}=={dist.version}")
    return packages


def main():
    report = []

    # ==================== SISTEMA ====================
    report.append("=== SISTEMA OPERACIONAL ===")
    report.append(f"Sistema: {platform.system()} {platform.release()}")
    report.append(f"Versão: {platform.version()}")
    report.append(f"Arquitetura: {platform.machine()}")
    report.append(f"Processador: {platform.processor()}")
    report.append(f"Python: {platform.python_version()}")

    # Distribuição Linux
    if platform.system() == "Linux":
        report.append(f"Distribuição: {run_cmd('lsb_release -d')}")
    report.append(
        f"RAM total: {round(psutil.virtual_memory().total / (1024**3), 2)} GB"
    )
    report.append(
        f"CPUs: {psutil.cpu_count(logical=False)} físicos / {psutil.cpu_count(logical=True)} lógicos"
    )

    # ==================== GPU ====================
    report.append("\n=== GPU (via nvidia-smi) ===")
    report.append(run_cmd("nvidia-smi"))

    report.append("\n=== GPU (via torch) ===")
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                report.append(
                    f"GPU {i}: {props.name} | Memória: {round(props.total_memory / 1024**3, 2)} GB | "
                    f"Multiprocessadores: {props.multi_processor_count}"
                )
        else:
            report.append("Nenhuma GPU CUDA detectada pelo PyTorch")
    except Exception as e:
        report.append(f"Erro ao consultar GPU pelo PyTorch: {e}")

    # CUDA versão do sistema
    report.append("\n=== CUDA (sistema) ===")
    report.append(run_cmd("nvcc --version"))

    # ==================== PYTORCH ====================
    report.append("\n=== PyTorch ===")
    try:
        report.append(f"torch.__version__: {torch.__version__}")
        report.append(f"CUDA disponível: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            report.append(f"Versão CUDA (PyTorch): {torch.version.cuda}")
            report.append(f"Versão cuDNN: {torch.backends.cudnn.version()}")
    except Exception as e:
        report.append(f"PyTorch não instalado ou erro ao carregar: {e}")

    # ==================== PACOTES ====================
    report.append("\n=== Pacotes relevantes instalados ===")
    packages = get_installed_packages()
    if packages:
        report.extend(packages)
    else:
        report.append("Nenhum pacote relevante encontrado")

    # ==================== RECOMENDAÇÕES ====================
    report.append("\n=== Recomendações ===")
    if "cpu" in torch.__version__:
        report.append(
            "⚠️ Você está usando PyTorch apenas com CPU. "
            "Se tiver GPU NVIDIA, instale versão com CUDA:\n"
            "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        )
    elif torch.cuda.is_available():
        report.append("✅ PyTorch está usando GPU corretamente")
    else:
        report.append(
            "⚠️ PyTorch não está enxergando a GPU. Verifique driver NVIDIA e versão instalada do PyTorch."
        )

    # Salvar em arquivo
    with open("diagnostico_pytorch.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("✅ Diagnóstico gerado em 'diagnostico_pytorch.txt'")


if __name__ == "__main__":
    main()
