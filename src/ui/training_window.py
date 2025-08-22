from PyQt5.QtCore import QProcess, pyqtSignal, QObject, Qt, QThread
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QProgressBar,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QMessageBox,
)
from PyQt5.QtGui import QFont
import logging
import sys
import os
import gc
import torch

logger = logging.getLogger(__name__)


class TrainingWorker(QObject):
    update_console = pyqtSignal(str)
    training_finished = pyqtSignal(bool, str)
    progress_updated = pyqtSignal(int, str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.process = None

    def start_training(self):
        try:
            python_exec = sys.executable
            script_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "train.py")
            )
            args = [
                f"--model={self.cfg.model}",
                f"--epochs={self.cfg.epochs}",
                f"--imgsz={self.cfg.imgsz}",
                f"--batch={self.cfg.batch}",
                f"--project={self.cfg.project}",
                f"--name={self.cfg.name}",
                f"--yaml_path={self.cfg.yaml_path}",
                f"--patience={self.cfg.patience}",
                f"--output_model={self.cfg.output_model}",
                # Remove this line: "--auto_augment",
            ]

            if self.cfg.reset:
                args.append("--reset")

            self.process = QProcess()
            # Junta stdout + stderr no mesmo canal
            self.process.setProcessChannelMode(QProcess.MergedChannels)

            self.process.readyReadStandardOutput.connect(self.handle_output)
            self.process.readyReadStandardError.connect(self.handle_error)  # <- stderr
            self.process.finished.connect(self.on_process_finished)

            # Garantir que o working dir seja a raiz do projeto
            root_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            self.process.setWorkingDirectory(root_dir)

            self.process.start(python_exec, [script_path] + args)

            if not self.process.waitForStarted():
                raise Exception("Falha ao iniciar o processo de treinamento")

        except Exception as e:
            logger.error(f"Erro no treinamento: {str(e)}", exc_info=True)
            self.training_finished.emit(False, f"Erro no treinamento: {str(e)}")

    def handle_output(self):
        output = (
            self.process.readAllStandardOutput().data().decode("utf-8", errors="ignore")
        )
        if output:
            self.update_console.emit(output)
            self.process_output(output)

    def handle_error(self):
        error_output = self.process.readAllStandardError().data().decode("utf-8")
        if error_output:
            self.update_console.emit(error_output)
            logger.error(error_output)

    def process_output(self, output):
        if "epoch" in output.lower() and "/" in output:
            try:
                parts = output.split()
                epoch_part = [p for p in parts if "/" in p][0]
                current, total = map(int, epoch_part.split("/"))
                progress = int((current / total) * 100)
                status = f"Época {current}/{total}"
                self.progress_updated.emit(progress, status)
            except Exception as e:
                logger.debug(f"Erro ao processar progresso: {str(e)}")

    def on_process_finished(self, exit_code, exit_status):
        if exit_code == 0 and exit_status == QProcess.NormalExit:
            self.training_finished.emit(True, "Treinamento concluído com sucesso!")
        else:
            error_msg = f"Processo terminou com código {exit_code}"
            if exit_status == QProcess.CrashExit:
                error_msg = "Processo terminou abruptamente"
            self.training_finished.emit(False, error_msg)

        self.cleanup()

    def cleanup(self):
        if self.process:
            self.process.kill()
            self.process.waitForFinished()
            self.process.deleteLater()
            self.process = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def stop(self):
        if self.process and self.process.state() == QProcess.Running:
            self.process.terminate()
            self.process.waitForFinished()


class TrainingWindow(QWidget):
    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("Progresso do Treinamento")
        self.setWindowModality(Qt.ApplicationModal)
        self.resize(800, 500)
        self.setup_ui()
        self.setup_thread()

    def setup_ui(self):
        layout = QVBoxLayout()

        header = QLabel("Treinamento do Modelo YOLOv8")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Inicializando treinamento...")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        layout.addWidget(self.console)

        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.clicked.connect(self.cancel_training)
        layout.addWidget(self.cancel_btn)

        self.setLayout(layout)

    def setup_thread(self):
        self.thread = QThread()
        self.worker = TrainingWorker(self.cfg)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.start_training)
        self.worker.update_console.connect(self.console.appendPlainText)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.training_finished.connect(self.on_training_finished)

        self.thread.start()

    def update_progress(self, progress, status):
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)

    def on_training_finished(self, success, message):
        if success:
            QMessageBox.information(self, "Sucesso", message)
        else:
            QMessageBox.critical(self, "Erro", message)
        self.close()

    def cancel_training(self):
        reply = QMessageBox.question(
            self,
            "Confirmar",
            "Deseja realmente cancelar o treinamento?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.worker.stop()
            self.thread.quit()
            self.thread.wait()
            self.close()

    def closeEvent(self, event):
        if self.thread.isRunning():
            self.cancel_training()
            event.ignore()
        else:
            event.accept()
