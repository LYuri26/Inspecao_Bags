from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtMultimedia import QSound
from PyQt5.QtWidgets import QLabel
import os
import time
import logging

logger = logging.getLogger(__name__)


class SoundHandler(QObject):
    # Sinal para tocar som na thread principal
    play_sound_signal = pyqtSignal()

    def __init__(
        self,
        alert_layout,
        alert_panel,
        results_label,
        sound_cooldown_secs=2,
        parent=None,
    ):
        super().__init__(parent)
        self.alert_layout = alert_layout  # layout onde ficam os alertas QLabel
        self.alert_panel = (
            alert_panel  # scroll area dos alertas (para scroll automático)
        )
        self.results_label = results_label  # QLabel para mostrar resultados (OK / Erro)
        self.sound_cooldown_secs = sound_cooldown_secs

        self.last_sound_time = 0
        self.defect_last_triggered = {}

        self.sound_enabled = False
        self.defect_sound = None

        self._configure_sound()

        # Conecta o sinal para tocar som
        self.play_sound_signal.connect(self._play_sound)

    def _configure_sound(self):
        root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        )
        sounds_path = os.path.join(root, "assets", "sounds")
        defect_sound_path = os.path.join(sounds_path, "defect_alert.wav")

        logger.info(f"Caminho do som: {defect_sound_path}")

        if not os.path.exists(defect_sound_path):
            logger.warning(f"⚠️ Arquivo de som não encontrado: {defect_sound_path}")
            self.sound_enabled = False
        else:
            self.defect_sound = QSound(defect_sound_path)
            self.sound_enabled = True

    def trigger_alert(self, message, defect_key=None):
        current_time = time.time()

        # cooldown por defeito
        if defect_key:
            last = self.defect_last_triggered.get(defect_key, 0)
            if current_time - last < self.sound_cooldown_secs:
                return  # ainda no cooldown
            self.defect_last_triggered[defect_key] = current_time

        self.add_alert(message, severity=2)

        if self.sound_enabled:
            if current_time - self.last_sound_time >= self.sound_cooldown_secs:
                self.last_sound_time = current_time
                # Emitir sinal para tocar som na thread principal
                self.play_sound_signal.emit()

    def _play_sound(self):
        logger.info("Tocando som de alerta")
        if self.defect_sound:
            self.defect_sound.play()

    def add_alert(self, message, severity=2):
        alert = QLabel(message, wordWrap=True)
        alert.setMargin(5)
        style_map = {
            1: "background-color: #FFF3CD; border-left: 4px solid #FFC107;",
            2: "background-color: #F8D7DA; border-left: 4px solid #DC3545;",
            3: "background-color: #D4EDDA; border-left: 4px solid #28A745;",
        }
        alert.setStyleSheet(style_map.get(severity, style_map[2]))
        self.alert_layout.insertWidget(0, alert)  # adiciona no topo da lista
        self.alert_panel.ensureWidgetVisible(alert)
