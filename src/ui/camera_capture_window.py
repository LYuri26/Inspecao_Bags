import cv2
import time
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget,
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QMessageBox,
    QVBoxLayout,
    QScrollArea,
    QProgressBar,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from .views.camera_module.camera_manager import CameraManager


class CaptureThread(QThread):
    update_progress = pyqtSignal(int, int, str)  # current, total, status
    finished_capture = pyqtSignal(list)

    def __init__(
        self, camera_manager, output_dir, num_cameras, num_photos=100, interval=5
    ):
        super().__init__()
        self.camera_manager = camera_manager
        self.output_dir = output_dir
        self.num_cameras = num_cameras
        self.num_photos = num_photos
        self.interval = interval
        self._is_running = True

    def run(self):
        saved_files = []
        try:
            for photo_num in range(1, self.num_photos + 1):
                if not self._is_running:
                    break

                self.update_progress.emit(
                    photo_num,
                    self.num_photos,
                    f"Capturando foto {photo_num}/{self.num_photos}",
                )
                time.sleep(self.interval)  # Intervalo entre fotos

                timestamp = int(time.time())
                for cam_id in range(self.num_cameras):
                    frame = None

                    rtsp_url = None
                    if 0 <= cam_id < len(self.camera_manager.camera_urls):
                        rtsp_url = self.camera_manager.camera_urls[cam_id]

                    if rtsp_url:
                        # Usa RTSP em alta resolução
                        cap = cv2.VideoCapture(rtsp_url)
                        if cap.isOpened():
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
                            ret, highres_frame = cap.read()
                            if ret:
                                frame = highres_frame
                            cap.release()
                    else:
                        # Nenhuma URL configurada → pula esta câmera
                        continue

                    if frame is None:
                        frame = self.camera_manager.get_latest_frame(cam_id)

                    if frame is not None:
                        filename = f"cam{cam_id}_{timestamp}_{photo_num}.jpg"
                        filepath = self.output_dir / filename
                        try:
                            cv2.imwrite(str(filepath), frame)
                            saved_files.append(str(filepath))
                        except Exception:
                            continue

            self.finished_capture.emit(saved_files)
        except Exception as e:
            self.update_progress.emit(0, 0, f"Erro: {str(e)}")

    def stop(self):
        self._is_running = False


class CameraCaptureWindow(QWidget):
    def __init__(self, num_cameras: int = 9, start_menu=None):
        super().__init__()
        self.start_menu = start_menu
        self.setWindowTitle("Captura de Fotos Automática")
        self.resize(1000, 700)

        self.num_cameras = max(1, min(9, int(num_cameras)))
        self.latest_frames = {i: None for i in range(self.num_cameras)}
        self.capture_thread = None

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)

        self.btn_capture = QPushButton(
            "Iniciar Captura Automática (100 fotos a cada 5s)"
        )
        self.btn_capture.setMinimumHeight(40)
        self.btn_capture.clicked.connect(self.start_auto_capture)
        main_layout.addWidget(self.btn_capture)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_label = QLabel("Pronto para iniciar captura")
        self.progress_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.progress_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.grid_layout = QGridLayout(container)
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

        self.camera_labels = {}
        cols = 3
        for i in range(self.num_cameras):
            lbl = QLabel(f"Cam {i}")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background-color: black; color: gray;")
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl.setMinimumSize(200, 150)
            self.grid_layout.addWidget(lbl, i // cols, i % cols)
            self.camera_labels[i] = lbl

        project_root = Path(__file__).resolve().parents[2]
        self.output_dir = project_root / "dataset_sacolas" / "baixadas"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.camera_manager = CameraManager(num_cameras=self.num_cameras)
        self.camera_manager.frame_ready.connect(self.on_frame_ready)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_display)
        self.timer.start(100)

        self.start_cameras()
        self.setLayout(main_layout)

    def start_cameras(self):
        try:
            self.camera_manager.start_capture()
        except Exception as e:
            QMessageBox.warning(self, "Erro", f"Falha ao iniciar câmeras: {e}")

    @pyqtSlot(int, object)
    def on_frame_ready(self, camera_id, frame):
        if 0 <= camera_id < self.num_cameras:
            self.latest_frames[camera_id] = frame

    def refresh_display(self):
        for cam_id, lbl in self.camera_labels.items():
            frame = self.latest_frames.get(cam_id)
            if frame is None:
                lbl.setText(f"Cam {cam_id}\nAguardando...")
                continue
            try:
                small_frame = cv2.resize(frame, (320, 240))
                rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                lbl.setPixmap(QPixmap.fromImage(qimg))
            except Exception:
                continue

    def start_auto_capture(self):
        if self.capture_thread and self.capture_thread.isRunning():
            return

        self.btn_capture.setEnabled(False)
        self.capture_thread = CaptureThread(
            self.camera_manager,
            self.output_dir,
            self.num_cameras,
            num_photos=100,
            interval=5,
        )
        self.capture_thread.update_progress.connect(self.update_progress)
        self.capture_thread.finished_capture.connect(self.on_capture_finished)
        self.capture_thread.start()

    def update_progress(self, current, total, status):
        progress = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)
        self.progress_label.setText(status)

    def on_capture_finished(self, saved_files):
        self.btn_capture.setEnabled(True)
        self.progress_label.setText("Captura concluída!")
        QMessageBox.information(
            self,
            "Captura Concluída",
            f"Total de {len(saved_files)} imagens salvas no diretório:\n{self.output_dir}",
        )

    def capture_all_photos(self):
        timestamp = int(time.time())
        saved = []
        for cam_id, frame in self.latest_frames.items():
            if frame is None:
                continue

            rtsp_url = None
            if 0 <= cam_id < len(self.camera_manager.camera_urls):
                rtsp_url = self.camera_manager.camera_urls[cam_id]

            if rtsp_url:
                cap = cv2.VideoCapture(rtsp_url)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
                    ret, highres_frame = cap.read()
                    if ret:
                        frame = highres_frame
                    cap.release()
            else:
                continue

            filename = f"cam{cam_id}_{timestamp}.jpg"
            filepath = self.output_dir / filename
            try:
                cv2.imwrite(str(filepath), frame)
                saved.append(str(filepath))
            except Exception:
                continue

        if saved:
            QMessageBox.information(
                self, "Captura concluída", f"Imagens salvas:\n" + "\n".join(saved)
            )
        else:
            QMessageBox.warning(self, "Aviso", "Nenhuma câmera ativa para capturar.")

    def closeEvent(self, event):
        try:
            if self.capture_thread and self.capture_thread.isRunning():
                self.capture_thread.stop()
                self.capture_thread.wait()
            self.camera_manager.stop_capture()
        except Exception:
            pass

        if self.start_menu:
            self.start_menu.show()
        event.accept()
