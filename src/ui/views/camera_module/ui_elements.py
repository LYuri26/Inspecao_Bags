from PyQt5.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QComboBox,
    QWidget,
    QCompleter,
    QMainWindow,
    QMessageBox,
    QFrame,
    QHBoxLayout,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QPalette, QColor, QImage, QPixmap
from .utils import get_main_window
import cv2
import numpy as np
import logging
from PyQt5 import QtGui
import os
import time

logger = logging.getLogger(__name__)


class BaseCameraUI(QWidget):
    """Base class that provides common UI elements for camera views with modern design"""

    inspection_requested = pyqtSignal(object)
    camera_state_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image = None
        self.active_company = None
        self.setup_common_ui()
        self.setup_styles()

    def setup_styles(self):
        """Setup modern styling for the UI"""
        self.setStyleSheet(
            """
            QWidget {
                background-color: #f5f5f5;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QComboBox {
                min-height: 32px;
                padding: 5px 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background: white;
                font-size: 14px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                border-left: 1px solid #ddd;
            }
            QLabel#statusLabel {
                color: #555;
                font-weight: 500;
                padding: 5px;
                background: #e9e9e9;
                border-radius: 4px;
            }
            QLabel#resultsLabel {
                color: #333;
                font-weight: 500;
                padding: 8px;
                background: #e9f7ef;
                border-radius: 4px;
                border: 1px solid #d4edda;
            }
            QScrollArea#alertPanel {
                border: 1px solid #ddd;
                border-radius: 4px;
                background: white;
            }
            QScrollArea#alertPanel QWidget {
                background: transparent;
            }
            QLabel#cameraLabel {
                background-color: #000;
                border-radius: 4px;
                border: 1px solid #ddd;
            }
        """
        )

    def setup_common_ui(self):
        """Setup common UI elements with modern layout"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Header section
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        # Company selection
        self.setup_company_combo()
        header_layout.addWidget(self.company_combo)

        # Status label
        self.status_label = QLabel("Status: Aguardando inicialização")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setObjectName("statusLabel")
        header_layout.addWidget(self.status_label)

        layout.addWidget(header_widget, stretch=0)

        # Main content area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Camera display
        self.camera_display = self.create_camera_display()
        content_layout.addWidget(self.camera_display, stretch=1)

        layout.addWidget(content_widget, stretch=1)

        # Footer section
        footer_widget = QWidget()
        footer_layout = QVBoxLayout(footer_widget)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(8)

        # Alert panel
        self.setup_alert_panel()
        footer_layout.addWidget(self.alert_panel, stretch=1)

        # Results label
        self.results_label = QLabel()
        self.results_label.setAlignment(Qt.AlignCenter)
        self.results_label.setWordWrap(True)
        self.results_label.setObjectName("resultsLabel")

        layout.addWidget(footer_widget, stretch=0)

    def setup_company_combo(self):
        """Setup company combobox with search functionality"""
        self.company_combo = QComboBox()
        self.company_combo.setEditable(True)
        self.company_combo.setInsertPolicy(QComboBox.NoInsert)
        self.company_combo.lineEdit().setPlaceholderText("Pesquisar empresa...")
        self.company_combo.lineEdit().setClearButtonEnabled(True)
        self.company_combo.completer().setCompletionMode(QCompleter.PopupCompletion)
        self.company_combo.completer().setFilterMode(Qt.MatchContains)
        self.company_combo.completer().setCaseSensitivity(Qt.CaseInsensitive)
        self.company_combo.setObjectName("companyCombo")

    def setup_alert_panel(self):
        """Setup alert panel with scrollable content"""
        self.alert_panel = QScrollArea()
        self.alert_panel.setWidgetResizable(True)
        self.alert_panel.setFrameShape(QScrollArea.NoFrame)
        self.alert_panel.setObjectName("alertPanel")
        self.alert_panel.setMinimumHeight(120)
        self.alert_panel.setMaximumHeight(200)

        self.alert_content = QWidget()
        self.alert_layout = QVBoxLayout(self.alert_content)
        self.alert_layout.setAlignment(Qt.AlignTop)
        self.alert_layout.setSpacing(6)
        self.alert_panel.setWidget(self.alert_content)

    def create_camera_display(self):
        """Creates the camera display widget"""
        label = QLabel(alignment=Qt.AlignCenter)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        label.setObjectName("cameraLabel")
        return label

    def update_grid_display(self, frames, cols=3, rows=3):
        if not frames or len(frames) != cols * rows:
            logger.warning(f"Invalid frames list: {len(frames) if frames else 'None'}")
            return

        container = self.camera_display
        if not isinstance(container, QLabel):
            return

        container_width = container.width()
        container_height = container.height()

        if container_width == 0 or container_height == 0:
            return

        # Calculate thumbnail size based on container dimensions
        padding = 4  # Minimal padding between frames
        thumb_w = (container_width - (cols - 1) * padding) // cols
        thumb_h = (container_height - (rows - 1) * padding) // rows

        # Minimum size for thumbnails to maintain visibility
        min_thumb_size = 200  # Minimum size for each thumbnail
        if thumb_w < min_thumb_size or thumb_h < min_thumb_size:
            # If thumbnails would be too small, reduce the padding
            padding = max(2, padding - 1)
            thumb_w = (container_width - (cols - 1) * padding) // cols
            thumb_h = (container_height - (rows - 1) * padding) // rows

        processed_frames = []
        for frame in frames:
            if frame is not None:
                h, w = frame.shape[:2]

                # Calculate scaling while maintaining aspect ratio
                scale_w = thumb_w / w
                scale_h = thumb_h / h
                scale = min(scale_w, scale_h)

                new_w = max(int(w * scale), 1)
                new_h = max(int(h * scale), 1)

                # Resize with high-quality interpolation
                resized = cv2.resize(
                    frame, (new_w, new_h), interpolation=cv2.INTER_AREA
                )

                # Add black borders if needed to maintain grid alignment
                delta_w = max(thumb_w - new_w, 0)
                delta_h = max(thumb_h - new_h, 0)
                top = delta_h // 2
                bottom = delta_h - top
                left = delta_w // 2
                right = delta_w - left

                resized_frame = cv2.copyMakeBorder(
                    resized,
                    top,
                    bottom,
                    left,
                    right,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
            else:
                # Create placeholder with dynamic text size
                placeholder_text = "SEM IMAGEM"
                font_scale = max(0.5, min(1.5, thumb_w / 300))  # Dynamic font scaling
                resized_frame = self.create_placeholder_image(
                    placeholder_text, (thumb_w, thumb_h), font_scale
                )

            processed_frames.append(resized_frame)

        # Build grid with padding between images
        rows_list = []
        for r in range(rows):
            row_frames = processed_frames[r * cols : (r + 1) * cols]
            row_image = row_frames[0]
            for f in row_frames[1:]:
                row_image = np.hstack(
                    [row_image, np.zeros((thumb_h, padding, 3), dtype=np.uint8), f]
                )
            rows_list.append(row_image)

        grid_image = rows_list[0]
        for r_img in rows_list[1:]:
            grid_image = np.vstack(
                [
                    grid_image,
                    np.zeros((padding, grid_image.shape[1], 3), dtype=np.uint8),
                    r_img,
                ]
            )

        # Convert to QImage and display
        height, width, channel = grid_image.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(
            grid_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
        ).rgbSwapped()

        container.setPixmap(QtGui.QPixmap.fromImage(q_img))
        self.current_image = grid_image

    def create_placeholder_image(
        self, text="SEM IMAGEM", size=(640, 480), font_scale=1.0
    ):
        """Creates a placeholder image with dynamic text size"""
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = max(1, int(font_scale * 2))
        textsize = cv2.getTextSize(text, font, font_scale, thickness)[0]
        textX = (img.shape[1] - textsize[0]) // 2
        textY = (img.shape[0] + textsize[1]) // 2
        cv2.putText(
            img,
            text,
            (textX, textY),
            font,
            font_scale,
            (0, 0, 255),
            thickness,
            cv2.LINE_AA,
        )
        return img

    def set_companies(self, companies):
        """Configures the company list in the combobox"""
        self.company_combo.clear()
        self.company_combo.addItem("Selecione uma empresa", None)

        for company in companies:
            self.company_combo.addItem(company["name"], company)

        try:
            self.company_combo.currentIndexChanged.disconnect()
        except TypeError:
            pass

        self.company_combo.currentIndexChanged.connect(self.on_company_changed)

        # Reconnect main window handler if exists
        if main_window := get_main_window(self):
            if hasattr(main_window, "on_company_changed"):
                self.company_combo.currentIndexChanged.connect(
                    main_window.on_company_changed
                )

    def on_company_changed(self, index):
        """Called when the selected company changes"""
        if hasattr(self, "company_combo") and self.company_combo is not None:
            company = self.company_combo.itemData(index) if index > 0 else None
            self.active_company = company
            logger.info(f"Empresa ativa alterada para: {company}")
            self.update_status(
                f"Empresa selecionada: {company['name']}"
                if company
                else "Nenhuma empresa selecionada"
            )
        else:
            logger.warning("company_combo não foi inicializado corretamente.")

    def update_status(self, message):
        """Updates the status label with a message"""
        self.status_label.setText(f"Status: {message}")

    def add_alert(self, message, alert_type="info"):
        """Adds an alert message to the alert panel"""
        color_map = {
            "info": "#d1ecf1",
            "warning": "#fff3cd",
            "error": "#f8d7da",
            "success": "#d4edda",
        }

        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(
            f"""
            background-color: {color_map.get(alert_type, '#f8f9fa')};
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 6px;
        """
        )

        label = QLabel(message)
        label.setWordWrap(True)

        layout = QHBoxLayout(frame)
        layout.addWidget(label)

        self.alert_layout.addWidget(frame)
        self.alert_panel.verticalScrollBar().setValue(
            self.alert_panel.verticalScrollBar().maximum()
        )

    def clear_alerts(self):
        """Clears all alerts from the alert panel"""
        while self.alert_layout.count():
            item = self.alert_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def display_results(self, message, result_type="info"):
        """Displays results in the results label"""
        color_map = {
            "info": "#d1ecf1",
            "warning": "#fff3cd",
            "error": "#f8d7da",
            "success": "#d4edda",
        }

        self.results_label.setText(message)
        self.results_label.setStyleSheet(
            f"""
            background-color: {color_map.get(result_type, '#f8f9fa')};
            border: 1px solid {color_map.get(result_type, '#ddd')};
        """
        )


class MonitorWindow(QMainWindow):
    def __init__(self, parent=None, model=None, active_company=None):
        super().__init__(parent=None, flags=Qt.Window)
        self.model = model
        self.active_company = active_company
        self.camera_view = None
        self.setup_ui()

    def setup_ui(self):
        """Configure modern responsive monitoring window"""
        self.setWindowTitle(
            f"Monitoramento - {self.active_company['name']}"
            if self.active_company
            else "Monitoramento"
        )

        # Main window styling
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f0f2f5;
            }
        """
        )

        # Central widget with modern layout
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        central_widget.setStyleSheet(
            """
            QWidget#centralWidget {
                background-color: #f5f5f5;
                border-radius: 8px;
            }
        """
        )
        self.setCentralWidget(central_widget)

        # Main layout with proper spacing
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Create camera view with error handling
        try:
            from .camera_view import CameraView

            self.camera_view = CameraView(self, self.model)

            if not hasattr(self.camera_view, "create_camera_display"):
                raise AttributeError("CameraView não implementa create_camera_display")

            self.camera_view.setMinimumSize(800, 600)

            # Add camera view to main layout
            main_layout.addWidget(self.camera_view, stretch=1)

            # Set initial company if available
            if self.active_company:
                self.camera_view.set_companies([self.active_company])
                index = self.camera_view.company_combo.findText(
                    self.active_company["name"]
                )
                if index >= 0:
                    self.camera_view.company_combo.setCurrentIndex(index)

        except Exception as e:
            logger.error(f"Falha ao inicializar visualização da câmera: {str(e)}")
            error_widget = QLabel("Falha ao inicializar o sistema de câmeras")
            error_widget.setAlignment(Qt.AlignCenter)
            error_widget.setStyleSheet("color: red; font-size: 16px;")
            main_layout.addWidget(error_widget)

        # Window size and position
        screen = self.screen().availableGeometry()
        initial_width = min(int(screen.width() * 0.9), 1400)
        initial_height = min(int(screen.height() * 0.85), 900)
        self.resize(initial_width, initial_height)

        if self.parent():
            self.move(self.parent().geometry().center() - self.rect().center())

    def resizeEvent(self, event):
        """Responsive resizing with better handling"""
        super().resizeEvent(event)

        if hasattr(self, "camera_view") and self.camera_view:
            if hasattr(self.camera_view, "update_frame"):
                self.camera_view.update_frame()

        # Dynamic font scaling
        font = self.font()
        base_size = max(10, int(self.width() / 100))
        font.setPointSize(base_size)
        self.setFont(font)

    def closeEvent(self, event):
        # Evita que seja chamado por eventos externos
        if self.isVisible():
            reply = QMessageBox.question(
                self,
                "Fechar Monitoramento",
                "Deseja realmente fechar a janela de monitoramento?\nIsso irá parar todas as câmeras.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                self.cleanup()
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()

    def cleanup(self):
        """Clean up when closing window"""
        if hasattr(self, "camera_view") and self.camera_view:
            if hasattr(self.camera_view, "deactivate_camera"):
                self.camera_view.deactivate_camera()
            self.camera_view = None

    def sync_with_main_window(self, main_window):
        """Sync with main window"""
        if not hasattr(main_window, "views") or "camera" not in main_window.views:
            return

        main_camera_view = main_window.views["camera"]

        if hasattr(main_camera_view, "company_combo"):
            self.sync_company_combo(main_camera_view.company_combo)

        if hasattr(self.camera_view, "activate_camera"):
            self.camera_view.activate_camera()

    def sync_company_combo(self, main_combo):
        """Sync company combobox"""
        if not hasattr(self.camera_view, "company_combo"):
            return

        self.camera_view.company_combo.clear()
        for i in range(main_combo.count()):
            text = main_combo.itemText(i)
            data = main_combo.itemData(i)
            self.camera_view.company_combo.addItem(text, data)

        if self.active_company:
            index = self.camera_view.company_combo.findText(self.active_company["name"])
            if index >= 0:
                self.camera_view.company_combo.setCurrentIndex(index)
                self.camera_view.active_company = self.active_company
