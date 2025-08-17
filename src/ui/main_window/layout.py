from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QSizePolicy,
    QStackedWidget,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication
from src.ui.widgets.sidebar import Sidebar


class MainLayout(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("mainLayout")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Obter informações da tela
        screen = QGuiApplication.primaryScreen().availableGeometry()

        # Layout principal
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.setLayout(self.main_layout)

        # Sidebar com largura adaptativa
        self.sidebar = Sidebar()
        self.sidebar.setObjectName("sidebar")

        # Define largura entre 200px (mínimo) e 15% da tela (máximo)
        sidebar_width = max(200, min(int(screen.width() * 0.15), 300))
        self.sidebar.setFixedWidth(sidebar_width)
        self.main_layout.addWidget(self.sidebar)

        # Área de conteúdo principal
        self.content_area = QStackedWidget()
        self.content_area.setObjectName("contentArea")
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout.addWidget(self.content_area, stretch=1)

        # Tamanho mínimo baseado na tela
        min_width = max(1024, int(screen.width() * 0.4))
        min_height = max(768, int(screen.height() * 0.5))
        self.setMinimumSize(min_width, min_height)
