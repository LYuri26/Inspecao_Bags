import sys
import subprocess
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QApplication,
    QMessageBox,
    QSizePolicy,
    QLabel,
    QSizeGrip,
    QFrame,
)
from PyQt5.QtCore import Qt, QSize
import logging
from PyQt5.QtGui import QFont, QGuiApplication, QIcon, QColor
from .training_window import TrainingWindow
from train import configurar_argumentos
from src.ui.main_window.window import MainWindow
from src.detector.detector import BagDetector
from .camera_capture_window import CameraCaptureWindow

logger = logging.getLogger(__name__)


class StartMenu(QWidget):
    """
    Menu inicial completo do sistema de inspeção de sacolas com:
    - Início do sistema
    - Treinamento de IA
    - Captura de fotos
    - Verificação inteligente de modelos
    - Totalmente adaptável para telas de alta resolução (4K/8K)
    """

    def __init__(self):
        super().__init__()
        self.configure_dpi_settings()
        self.setWindowTitle("Menu Inicial - Sistema de Inspeção de Sacolas")
        self.setWindowIcon(QIcon("assets/icons/app_icon.png"))
        self.setup_responsive_layout()
        self.setup_ui()
        self.setup_styles()

    def configure_dpi_settings(self):
        """Configuração para alta resolução"""
        QGuiApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QGuiApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

        # Aumenta o tamanho base da fonte
        font = QGuiApplication.font()
        font.setPointSize(12)  # Tamanho base maior
        QGuiApplication.setFont(font)

    def setup_responsive_layout(self):
        """Ajusta o tamanho baseado na tela"""
        screen = QGuiApplication.primaryScreen().availableGeometry()
        width = min(max(800, int(screen.width() * 0.3)), 1400)
        height = min(max(600, int(screen.height() * 0.5)), 1000)
        self.resize(width, height)
        self.setMinimumSize(int(width * 0.7), int(height * 0.7))

    def setup_ui(self):
        """Configura a interface"""
        layout = QVBoxLayout()
        layout.setSpacing(30)
        layout.setContentsMargins(40, 40, 40, 40)

        # Título com fonte grande
        title = QLabel("Sistema de Inspeção de Sacolas")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont("Arial", 24, QFont.Bold)
        title.setFont(title_font)
        title.setStyleSheet("color: #2c3e50; margin-bottom: 30px;")
        layout.addWidget(title)

        # Container de botões
        button_frame = QFrame()
        button_layout = QVBoxLayout()
        button_layout.setSpacing(20)

        # Botões principais
        buttons = [
            ("▶ Iniciar Sistema", self.start_system, "#27ae60"),
            ("🤖 Treinar Modelo de IA", self.train_ai, "#f39c12"),
            ("📸 Capturar Fotos para Treinamento", self.capture_photos, "#8e44ad"),
        ]

        for text, callback, color in buttons:
            btn = QPushButton(text)
            btn.clicked.connect(callback)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            btn.setMinimumHeight(70)  # Altura fixa suficiente
            btn.setCursor(Qt.PointingHandCursor)

            # Estilo com bom contraste
            btn.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    border-radius: 10px;
                    padding: 15px;
                    font-size: 18px;
                    font-weight: bold;
                    border: 2px solid {self.darken_color(color, 10)};
                }}
                QPushButton:hover {{
                    background-color: {self.darken_color(color, 15)};
                    color: white;  /* Mantém texto branco no hover */
                }}
                QPushButton:pressed {{
                    background-color: {self.darken_color(color, 25)};
                }}
            """
            )

            button_layout.addWidget(btn)

        button_frame.setLayout(button_layout)
        layout.addWidget(button_frame)

        # Rodapé
        footer = QLabel(
            "Sistema de Inspeção Automática v1.0\n"
            "Modo tradicional: estabilidade > novidade"
        )
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet(
            """
            font-size: 14px; 
            color: #7f8c8d; 
            margin-top: 30px;
        """
        )
        layout.addWidget(footer)

        self.setLayout(layout)

    def darken_color(self, hex_color, percent):
        """Escurece uma cor HEX mantendo o contraste"""
        color = QColor(hex_color)
        return color.darker(100 + percent).name()

    def setup_styles(self):
        """Estilos gerais para melhor legibilidade"""
        self.setStyleSheet(
            """
            QWidget {
                background: #f5f7fa;
                font-family: 'Segoe UI', Arial;
            }
            QLabel {
                font-size: 16px;
            }
            QMessageBox {
                font-size: 16px;
            }
            QMessageBox QLabel {
                font-size: 16px;
            }
        """
        )

    def check_model_async(self):
        """Verifica o modelo de forma não bloqueante"""
        model_path = Path("modelos/detector_sacola.pt")
        if not model_path.exists():
            logger.info(
                "Modelo não encontrado - usuário será notificado ao tentar iniciar"
            )
            self.btn_start.setToolTip("Modelo não encontrado - treine primeiro")
            self.btn_start.setStyleSheet(
                self.btn_start.styleSheet()
                + """
                QPushButton {
                    border: 2px dashed #c0392b;
                }
            """
            )

    def start_system(self):
        """Inicia o sistema principal de inspeção"""
        try:
            model_path = Path("modelos/detector_sacola.pt")

            if not model_path.exists():
                self.handle_missing_model()
                return

            self.load_and_start_model(model_path)

        except Exception as e:
            logger.error(f"Falha ao iniciar sistema: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self,
                "Erro de Inicialização",
                f"Não foi possível iniciar o sistema:\n\n{str(e)}\n\n"
                "Verifique:\n"
                "1. Se o modelo existe\n"
                "2. Se o modelo é compatível\n"
                "3. Se há dispositivos de câmera conectados",
            )

    def load_and_start_model(self, model_path):
        """Carrega o modelo e inicia a janela principal com configuração de DPI"""
        try:
            model = BagDetector(str(model_path))

            # Garante que a janela principal também tenha scaling adequado
            from PyQt5.QtCore import Qt
            from PyQt5.QtGui import QGuiApplication

            QGuiApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
            QGuiApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

            self.main_window = MainWindow(model)
            self.main_window.show()
            self.close()
        except Exception as e:
            raise Exception(f"Falha ao carregar modelo: {str(e)}")

    def handle_missing_model(self):
        """Lida com a ausência do modelo de forma inteligente"""
        choice = QMessageBox.question(
            self,
            "Modelo Não Encontrado",
            "Nenhum modelo treinado foi encontrado.\n\n"
            "Deseja treinar um novo modelo agora?\n"
            "(Recomendado para primeira execução)",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes,
        )

        if choice == QMessageBox.Yes:
            self.train_ai()
        elif choice == QMessageBox.No:
            QMessageBox.information(
                self,
                "Instruções",
                "Para usar o sistema, você precisa:\n\n"
                "1. Treinar um modelo (opção 'Treinar Modelo de IA')\n"
                "2. Ou colocar um modelo pré-treinado em:\n"
                "   'modelos/detector_sacola.pt'\n\n"
                "O modelo deve estar no formato YOLOv8 (.pt)",
            )

    def train_ai(self):
        """Inicia o processo de treinamento"""
        try:
            cfg = configurar_argumentos()
            self.training_window = TrainingWindow(cfg)
            self.training_window.show()
            self.hide()
        except Exception as e:
            logger.error(f"Erro ao iniciar treinamento: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self,
                "Erro no Treinamento",
                f"Falha ao iniciar o treinamento:\n\n{str(e)}",
            )

    def on_training_finished(self, success, message):
        """Lida com o término do treinamento"""
        self.show()
        if success:
            choice = QMessageBox.question(
                self,
                "Treinamento Concluído",
                f"{message}\n\nDeseja iniciar o sistema agora?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if choice == QMessageBox.Yes:
                self.start_system()
        else:
            QMessageBox.critical(
                self, "Erro no Treinamento", f"O treinamento falhou:\n\n{message}"
            )

    def capture_photos(self):
        """Abre a janela de captura de fotos"""
        try:
            self.capture_window = CameraCaptureWindow(start_menu=self)
            self.capture_window.show()
            self.hide()
        except Exception as e:
            logger.error(f"Erro ao abrir captura: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self,
                "Erro de Câmera",
                f"Não foi possível iniciar a captura:\n\n{str(e)}\n\n"
                "Verifique:\n"
                "1. Se as câmeras estão conectadas\n"
                "2. Se os drivers estão instalados",
            )


if __name__ == "__main__":
    # Configuração de DPI para execução direta
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QGuiApplication

    QGuiApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QGuiApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)

    # Configuração de fonte escalável
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)

    window = StartMenu()
    window.show()
    sys.exit(app.exec_())
