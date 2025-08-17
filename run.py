#!/usr/bin/env python3
import os
import sys
import platform
import subprocess
from pathlib import Path
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Configurações
VENV_DIR = ".venv"
REQUIREMENTS_FILE = "requirements.txt"
MAIN_SCRIPT = "src/main.py"  # Voltamos a usar main.py como entry point


def obter_caminho_python():
    """Retorna o caminho para o Python no ambiente virtual"""
    if platform.system() == "Windows":
        return Path(VENV_DIR) / "Scripts" / "python.exe"
    return Path(VENV_DIR) / "bin" / "python"


def criar_ambiente_virtual():
    """Cria o ambiente virtual se não existir"""
    if not Path(VENV_DIR).exists():
        logger.info("Criando ambiente virtual...")
        try:
            subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
            logger.info("✅ Ambiente virtual criado")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Falha ao criar ambiente virtual: {str(e)}")
            return False
    return True


def instalar_dependencias():
    """Instala as dependências do projeto"""
    logger.info("Instalando dependências...")

    pip_exec = "pip.exe" if platform.system() == "Windows" else "pip"
    pip_path = (
        Path(VENV_DIR) / "bin" / pip_exec
        if platform.system() != "Windows"
        else Path(VENV_DIR) / "Scripts" / pip_exec
    )

    try:
        subprocess.run([str(pip_path), "install", "-r", REQUIREMENTS_FILE], check=True)
        logger.info("✅ Dependências instaladas com sucesso")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Falha ao instalar dependências: {str(e)}")
        return False


def executar_aplicacao():
    """Inicia a aplicação principal"""
    python_path = obter_caminho_python()

    logger.info("Iniciando aplicação...")
    try:
        # Configura o ambiente e PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path.cwd())

        # Cria pastas necessárias
        Path("modelos").mkdir(exist_ok=True)
        Path("dataset_sacolas/baixadas").mkdir(parents=True, exist_ok=True)

        # Executa via main.py que importa corretamente os módulos
        subprocess.run([str(python_path), MAIN_SCRIPT], env=env, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Falha ao executar a aplicação: {str(e)}")
        return False


def mostrar_comando_manual():
    """Mostra o comando para execução manual"""
    logger.info("\n⚠️ Se necessário, execute manualmente com:")
    if platform.system() == "Windows":
        logger.info(f"  {VENV_DIR}\\Scripts\\activate && python {MAIN_SCRIPT}")
    else:
        logger.info(f"  source {VENV_DIR}/bin/activate && python {MAIN_SCRIPT}")


def main():
    try:
        logger.info("\n" + "=" * 60)
        logger.info(" SISTEMA DE INSPEÇÃO DE SACOLAS ".center(60, "="))
        logger.info("=" * 60 + "\n")

        if not criar_ambiente_virtual():
            return 1

        if not instalar_dependencias():
            return 1

        if not executar_aplicacao():
            mostrar_comando_manual()
            return 1

        logger.info("\n✅ Sistema executado com sucesso!")
        return 0

    except Exception as e:
        logger.error(f"❌ Erro fatal: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
