#!/usr/bin/env python3
import os
import sys
import platform
import subprocess
from pathlib import Path
import logging

# ================= CONFIGURAÇÃO =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

VENV_DIR = Path(".venv")
REQ_CPU = Path("requirements-cpu.txt")
REQ_GPU = Path("requirements-gpu.txt")
MAIN_SCRIPT = Path("src") / "main.py"


# ================= FUNÇÕES =================
def obter_caminho_python():
    """Retorna o caminho do Python no venv"""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def criar_ambiente_virtual():
    """Cria o venv se não existir"""
    if not VENV_DIR.exists():
        logger.info("⚙️ Criando ambiente virtual...")
        try:
            subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
            logger.info("✅ Ambiente virtual criado")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Falha ao criar ambiente virtual: {e}")
            return False
    else:
        logger.info("✅ Ambiente virtual já existe")
    return True


def instalar_dependencias():
    """Instala pacotes CPU ou GPU dependendo do hardware"""
    python_path = obter_caminho_python()
    pip_path = python_path.parent / (
        "pip.exe" if platform.system() == "Windows" else "pip"
    )

    # Atualiza pip primeiro
    try:
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
        logger.info("✅ pip atualizado com sucesso")
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️ Falha ao atualizar pip: {e}")

    # Verifica se há GPU NVIDIA disponível
    has_nvidia_gpu = False
    try:
        if platform.system() == "Windows":
            import winreg

            try:
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SYSTEM\CurrentControlSet\Services\nvlddmkm",
                )
                winreg.CloseKey(key)
                has_nvidia_gpu = True
            except:
                has_nvidia_gpu = False
        else:
            result = subprocess.run(
                ["which", "nvidia-smi"], capture_output=True, text=True
            )
            has_nvidia_gpu = result.returncode == 0
    except:
        has_nvidia_gpu = False

    # Escolhe requirements
    if has_nvidia_gpu and REQ_GPU.exists():
        req_file = REQ_GPU
        logger.info("✅ GPU NVIDIA detectada → Instalando pacotes GPU")
    elif REQ_CPU.exists():
        req_file = REQ_CPU
        logger.warning("⚠️ GPU NVIDIA não detectada → Instalando pacotes CPU")
    else:
        logger.error("❌ Nenhum arquivo de requirements encontrado")
        return False

    # Instala dependências
    try:
        subprocess.run([str(pip_path), "install", "-r", str(req_file)], check=True)
        logger.info(f"✅ Pacotes instalados com sucesso a partir de {req_file}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Falha ao instalar dependências: {e}")
        return False


def verificar_gpu():
    """Executa script dentro do venv para verificar GPU"""
    python_path = obter_caminho_python()
    try:
        subprocess.run(
            [
                str(python_path),
                "-c",
                (
                    "import torch; "
                    "print(f'🔥 Torch versão: {torch.__version__}'); "
                    "print(f'CUDA disponível: {torch.cuda.is_available()}'); "
                    "print(f'GPUs detectadas: {torch.cuda.device_count()}')"
                ),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erro ao verificar GPU: {e}")


def iniciar_main():
    """Executa o src/main.py dentro do venv"""
    python_path = obter_caminho_python()
    if MAIN_SCRIPT.exists():
        logger.info(f"🚀 Iniciando {MAIN_SCRIPT} ...")
        try:
            subprocess.run([str(python_path), "-m", "src.main"], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Erro ao rodar main: {e}")
    else:
        logger.warning(f"⚠️ Arquivo {MAIN_SCRIPT} não encontrado")


# ================= MAIN =================
def main():
    logger.info("=" * 60)
    logger.info(" INICIALIZADOR - AMBIENTE ".center(60, "="))
    logger.info("=" * 60)

    if not criar_ambiente_virtual():
        return 1

    if not instalar_dependencias():
        return 1

    verificar_gpu()
    iniciar_main()

    return 0


if __name__ == "__main__":
    sys.exit(main())
