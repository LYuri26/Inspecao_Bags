import sys
import os
import logging
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication
from src.ui.main_window.window import MainWindow
from src.detector.detector import BagDetector
from pathlib import Path
from src.ui.start_menu import StartMenu

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def configure_dpi():
    """Configura o scaling para alta DPI"""
    QGuiApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QGuiApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)


def load_styles():
    """Carrega os estilos CSS para a aplicação"""
    try:
        from assets.utils.styles import MAIN_STYLE

        return MAIN_STYLE
    except ImportError:
        logger.warning("Estilos não encontrados, usando padrão do sistema")
        return ""


def setup_environment():
    """Configurações iniciais com verificação de arquivos"""
    model_path = Path("modelos/detector_sacola.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    return str(model_path)


def main():
    try:
        # Configurar DPI antes de criar a QApplication
        configure_dpi()

        app = QApplication(sys.argv)

        # Configurar fonte base escalável
        font = app.font()
        font.setPointSize(10)  # Tamanho base que será escalado
        app.setFont(font)

        # Carregar estilos
        app.setStyleSheet(load_styles())

        # Criar e mostrar menu inicial
        start_menu = StartMenu()
        start_menu.show()

        sys.exit(app.exec_())

    except Exception as e:
        logger.error(f"Erro na aplicação: {str(e)}", exc_info=True)
        QMessageBox.critical(None, "Erro Fatal", f"Ocorreu um erro:\n{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
