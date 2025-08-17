from PyQt5.QtWidgets import (
    QMainWindow,
    QSizePolicy,
    QStatusBar,
    QLabel,
    QMessageBox,
    QWidget,
)
from PyQt5.QtCore import pyqtSignal, QTimer, Qt
from .layout import MainLayout
from src.ui.views.companies import CompaniesView
from src.ui.views.reports import ReportsView
from src.ui.views.history import HistoryView
from src.core.inspector import simple_classify, validate_detections
from assets.utils.styles import MAIN_STYLE
from src.ui.views.camera_module.camera_view import CameraView
from src.ui.views.camera_module.ui_elements import MonitorWindow
from PyQt5.QtGui import QGuiApplication

# Configurar High DPI scaling
QGuiApplication.setAttribute(Qt.AA_EnableHighDpiScaling)  # Habilita scaling automático
QGuiApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)  # Usa imagens de alta resolução


class MainWindow(QMainWindow):
    change_view_signal = pyqtSignal(str)
    inspection_results = pyqtSignal(dict)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.active_company = None

        # Configurações iniciais de DPI/Scaling
        self.configure_dpi_settings()

        self.setup_ui()
        self.adjust_font_sizes()  # Ajusta fontes após UI estar criada
        self.setup_connections()
        self.setup_camera_timer()
        self.setup_status_bar()

        self.companies_view = self.views["companies"]
        self.companies_view.load_companies_from_disk()

    def configure_dpi_settings(self):
        """Configura o comportamento de scaling para alta DPI"""
        QGuiApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QGuiApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

        # Configura fonte base para a aplicação
        font = self.font()
        font.setPointSize(10)  # Tamanho base que será escalado
        self.setFont(font)

    def setup_ui(self):
        """Configura a interface do usuário com dimensões adaptativas"""
        screen = QGuiApplication.primaryScreen().availableGeometry()

        # Tamanhos baseados na resolução da tela
        min_width = max(
            1024, int(screen.width() * 0.4)
        )  # Mínimo de 1024 ou 40% da tela
        min_height = max(
            600, int(screen.height() * 0.5)
        )  # Mínimo de 600 ou 50% da tela

        self.setWindowTitle("Sistema de Inspeção de Sacolas")
        self.setMinimumSize(min_width, min_height)
        self.resize(int(min_width * 1.2), int(min_height * 1.2))

        self.setWindowFlags(
            Qt.Window
            | Qt.CustomizeWindowHint
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )

        self.setStyleSheet(MAIN_STYLE)

        # Layout principal com dimensões adaptativas
        self.main_layout = MainLayout(self)
        self.setCentralWidget(self.main_layout)

        # Views com políticas de tamanho expansível
        self.views = {
            "companies": CompaniesView(self),
            "camera": CameraView(self, self.model),
            "reports": ReportsView(self),
            "history": HistoryView(self),
        }

        for name, view in self.views.items():
            view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.main_layout.content_area.addWidget(view)

    def adjust_font_sizes(self):
        """Ajusta os tamanhos de fonte baseado na resolução da tela"""
        screen = QGuiApplication.primaryScreen().availableGeometry()
        base_resolution = 1920  # Resolução base de referência
        scale_factor = max(
            1.0, min(screen.width() / base_resolution, 2.0)
        )  # Limita a 2x

        # Aplica ao widget principal
        font = self.font()
        base_size = 10  # Tamanho base em pontos
        font.setPointSize(int(base_size * scale_factor * 0.9))  # Ajuste fino
        self.setFont(font)

        def apply_scaling(widget):
            widget_font = widget.font()
            widget_font.setPointSize(int(base_size * scale_factor * 0.9))
            widget.setFont(widget_font)

            # Ajuste especial para QLabels
            if isinstance(widget, QLabel):
                widget.setStyleSheet(
                    f"font-size: {int(base_size * scale_factor * 0.9)}pt;"
                )

            # Propaga para filhos
            for child in widget.findChildren(QWidget):
                apply_scaling(child)

        apply_scaling(self)

    # Adicionar novo método para abrir detalhes da câmera
    def open_camera_detail(self, camera_id):
        """Abre uma janela de detalhes para a câmera específica"""
        detail_window = QMainWindow(self)
        detail_window.setWindowTitle(f"Câmera {camera_id+1} - Detalhes")

        # Cria uma nova instância de CameraView para o detalhe
        camera_view = CameraView(detail_window, self.model)
        camera_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Configurações adicionais
        detail_window.setCentralWidget(camera_view)
        detail_window.resize(800, 600)

        # Centraliza a janela
        detail_window.move(self.geometry().center() - detail_window.rect().center())

        # Inicia a câmera específica
        camera_view.start_camera(camera_id)
        detail_window.show()

    def setup_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.status_company = QLabel("Nenhuma empresa selecionada")
        self.status_camera = QLabel("Câmera: Desativada")
        self.status_inspection = QLabel("Inspeções: 0")

        self.status_bar.addPermanentWidget(self.status_company)
        self.status_bar.addPermanentWidget(QLabel("|"))
        self.status_bar.addPermanentWidget(self.status_camera)
        self.status_bar.addPermanentWidget(QLabel("|"))
        self.status_bar.addPermanentWidget(self.status_inspection)

    def setup_connections(self):
        """Configura todas as conexões de sinais para a janela principal"""
        # Conexões de navegação de view
        self.change_view_signal.connect(self.change_view)
        self.main_layout.sidebar.view_changed.connect(self.change_view_signal.emit)

        # Conexão para abrir janela de monitoramento
        self.main_layout.sidebar.open_monitor_window.connect(self.open_monitor_window)

        # Conexões de inspeção da câmera
        if "camera" in self.views:
            self.views["camera"].inspection_requested.connect(self.run_inspection)

            # Conexão para mudança de empresa
            if hasattr(self.views["camera"], "company_combo"):
                self.views["camera"].company_combo.currentIndexChanged.connect(
                    self.on_company_changed
                )

        # Conexão para empresa deletada
        if "companies" in self.views:
            self.views["companies"].company_deleted.connect(self.handle_company_deleted)

        # Conexão para resultados de inspeção
        self.inspection_results.connect(self.handle_inspection_results)

    def on_company_changed(self, index):
        """Método chamado quando a empresa selecionada na câmera muda"""
        camera_view_instance = self.views["camera"]
        if hasattr(camera_view_instance, "company_combo") and index >= 0:
            company_data = camera_view_instance.company_combo.itemData(index)

            # Update both the camera view and main window
            if hasattr(camera_view_instance, "active_company"):
                camera_view_instance.active_company = company_data

            # Update the main window's active company
            if company_data:
                self.set_active_company(company_data)
            else:
                self.active_company = None
                self.status_company.setText("Nenhuma empresa selecionada")

    def handle_company_deleted(self, company_name):
        """Lida com a exclusão de uma empresa"""
        if self.active_company and self.active_company["name"] == company_name:
            self.active_company = None
            self.setWindowTitle("Sistema de Inspeção de Sacolas")
            self.status_company.setText("Nenhuma empresa selecionada")
            self.views["companies"].load_companies_from_disk()
            self.views["reports"].load_companies([])

    def setup_camera_timer(self):
        self.camera_timer = QTimer()
        self.camera_timer.setInterval(30)

        # Não conectamos mais o timer diretamente aqui
        # A atualização de frames será tratada por cada CameraView individualmente

        # Conectamos apenas a mudança de estado
        if "camera" in self.views:
            self.views["camera"].camera_state_changed.connect(
                lambda state: (
                    self.camera_timer.start() if state else self.camera_timer.stop(),
                    self.status_camera.setText(
                        f"Câmera: {'Ativada' if state else 'Desativada'}"
                    ),
                )
            )

    def change_view(self, view_name: str):
        if view_name in self.views:
            self.main_layout.content_area.setCurrentWidget(self.views[view_name])

            if view_name == "camera":
                for i in range(9):
                    self.views["camera"].start_camera(i)
            else:
                self.views["camera"].stop_camera()

    def set_active_company(self, company):
        self.active_company = company
        if company:
            self.setWindowTitle(f"Sistema de Inspeção - {company['name']}")
            self.status_company.setText(f"Empresa: {company['name']}")

            # Atualiza lista interna e chama método sem parâmetro
            reports_view = self.views["reports"]
            reports_view.all_companies = [company["name"]]
            reports_view.load_companies()  # sem argumentos

            if "camera" in self.views and hasattr(
                self.views["camera"], "active_company"
            ):
                self.views["camera"].active_company = company
        else:
            self.setWindowTitle("Sistema de Inspeção de Sacolas")
            self.status_company.setText("Nenhuma empresa selecionada")

            if "camera" in self.views and hasattr(
                self.views["camera"], "active_company"
            ):
                self.views["camera"].active_company = None

    def load_companies(self):
        self.views["companies"].load_companies([])
        self.views["reports"].load_companies([])

    def run_inspection(self, image):
        try:
            processed_frame = self.model.process_frame(image)
            detections = self.model.detect(image)
            classified = simple_classify(detections)
            results = validate_detections(classified)
            self.inspection_results.emit(results)

            total = len(self.views["history"].model._data) + 1
            self.status_inspection.setText(f"Inspeções: {total}")

        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro na inspeção: {str(e)}")

    def handle_inspection_results(self, results):
        if results["valid_bag"]:
            if results["has_defects"]:
                self.views["camera"].display_results("Sacola com defeitos!")
                self.views["camera"].trigger_alert("Defeitos encontrados!")
            else:
                self.views["camera"].display_results("Sacola OK!")
        else:
            self.views["camera"].display_results("Sacola não detectada!")
            self.views["camera"].trigger_alert("Sacola inválida!")

        self.views["history"].add_inspection(
            {
                "empresa": (
                    self.active_company["name"] if self.active_company else "N/A"
                ),
                "resultado": (
                    "Aprovado"
                    if results["valid_bag"] and not results["has_defects"]
                    else "Reprovado"
                ),
                "defeitos": (
                    ", ".join(results["defects"]) if results["defects"] else "Nenhum"
                ),
                "confianca": f"{results['main_confidence']:.2f}",
            }
        )

    def open_monitor_window(self):
        """Abre uma nova janela independente para monitoramento"""
        monitor_window = MonitorWindow(
            parent=self, model=self.model, active_company=self.active_company
        )

        # Sincroniza com a janela principal
        monitor_window.sync_with_main_window(self)

        # Ajusta posição e exibe
        monitor_window.move(self.geometry().center() - monitor_window.rect().center())
        monitor_window.show()

        # Mantém referência para evitar garbage collection
        if not hasattr(self, "_monitor_windows"):
            self._monitor_windows = []
        self._monitor_windows.append(monitor_window)

        # Conexão para limpeza quando a janela for fechada
        monitor_window.destroyed.connect(
            lambda: (
                self._monitor_windows.remove(monitor_window)
                if hasattr(self, "_monitor_windows")
                else None
            )
        )

    def closeEvent(self, event):
        if "camera" in self.views:
            self.views["camera"].stop_camera()
        self.camera_timer.stop()
        super().closeEvent(event)
