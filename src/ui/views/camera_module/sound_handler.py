from PyQt5.QtCore import QObject, pyqtSignal, QTimer, QMutex, QThread
from PyQt5.QtMultimedia import QSound
import os
import time
import logging
from collections import deque, defaultdict
from PyQt5.QtWidgets import (
    QLabel,
)

logger = logging.getLogger(__name__)


class SoundWorker(QThread):
    """Thread dedicada para gerenciamento de sons"""

    finished = pyqtSignal()

    def __init__(self, sound_handler):
        super().__init__()
        self.sound_handler = sound_handler
        self.running = True

    def run(self):
        while self.running:
            self.sound_handler.process_queue()
            time.sleep(0.05)  # 50ms entre verificações
        self.finished.emit()


class SoundHandler(QObject):
    play_sound_signal = pyqtSignal(str, str, int)  # (sound_type, message, severity)

    def __init__(self, alert_layout, alert_panel, results_label, parent=None):
        super().__init__(parent)
        self.alert_layout = alert_layout
        self.alert_panel = alert_panel
        self.results_label = results_label

        # Configuração de som
        self.sounds = {}
        self.sound_queue = deque()
        self.currently_playing = None
        self.last_play_time = 0
        self.mutex = QMutex()
        self.sound_cooldown = 1.2  # Tempo mínimo entre sons (1.2 segundos)

        # Dicionário para evitar repetição excessiva do mesmo alerta
        self.last_alerts = defaultdict(lambda: 0)
        self.alert_cooldown = 3  # Segundos entre alertas do mesmo tipo

        # Thread de processamento
        self.worker = SoundWorker(self)
        self.worker.start()

        self._load_sounds()
        self.play_sound_signal.connect(self._add_to_queue)

    def _load_sounds(self):
        """Carrega todos os sons disponíveis"""
        sounds_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "assets", "sounds"
        )

        sound_files = {
            "defect": "defect_alert.wav",
            "warning": "warning.wav",
            "critical": "critical_alert.wav",
        }

        for sound_type, filename in sound_files.items():
            path = os.path.join(sounds_dir, filename)
            if os.path.exists(path):
                self.sounds[sound_type] = QSound(path)
                logger.info(f"Som carregado: {sound_type}")
            else:
                logger.warning(f"Arquivo de som não encontrado: {filename}")
                # Cria um som vazio para evitar erros
                self.sounds[sound_type] = QSound()

    def _add_to_queue(self, sound_type, message, severity):
        """Adiciona um som à fila de reprodução"""
        current_time = time.time()

        # Verifica cooldown para este tipo de alerta
        if current_time - self.last_alerts[sound_type] < self.alert_cooldown:
            return

        self.last_alerts[sound_type] = current_time

        self.mutex.lock()
        try:
            # Evita duplicatas consecutivas do mesmo som
            if not self.sound_queue or self.sound_queue[-1][0] != sound_type:
                self.sound_queue.append((sound_type, message, severity))
        finally:
            self.mutex.unlock()

    def process_queue(self):
        """Processa a fila de reprodução"""
        current_time = time.time()

        # Se está tocando algo, verifica se terminou
        if self.currently_playing:
            if current_time - self.last_play_time > self.sound_cooldown:
                self.currently_playing = None
            return

        if self.sound_queue:
            self.mutex.lock()
            try:
                if self.sound_queue:
                    sound_type, message, severity = self.sound_queue.popleft()
                    if sound_type in self.sounds:
                        self.currently_playing = sound_type
                        self.last_play_time = current_time

                        # Toca o som e mostra o alerta
                        self.sounds[sound_type].play()
                        self._show_alert(message, severity)
            finally:
                self.mutex.unlock()

    def _show_alert(self, message, severity):
        """Mostra o alerta na interface"""
        alert = QLabel(message)
        alert.setWordWrap(True)
        alert.setMargin(8)

        # Estilos baseados na severidade
        styles = {
            1: "background-color: #FFF3CD; border-left: 4px solid #FFC107; color: #856404;",
            2: "background-color: #F8D7DA; border-left: 4px solid #DC3545; color: #721C24;",
            3: "background-color: #D4EDDA; border-left: 4px solid #28A745; color: #155724;",
        }

        alert.setStyleSheet(styles.get(severity, styles[2]))
        self.alert_layout.insertWidget(0, alert)
        self.alert_panel.ensureWidgetVisible(alert)

    def trigger_alert(self, message, defect_type="defect", severity=2):
        """Dispara um alerta com gerenciamento de fila"""
        self.play_sound_signal.emit(defect_type, message, severity)

    def cleanup(self):
        """Limpeza ao encerrar"""
        self.worker.running = False
        self.worker.wait()

        for sound in self.sounds.values():
            sound.stop()
