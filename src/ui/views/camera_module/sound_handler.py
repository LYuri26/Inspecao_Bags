from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtMultimedia import QSound
import os
import time
import logging
from collections import deque
from PyQt5.QtWidgets import (
    QLabel,
)

logger = logging.getLogger(__name__)


class SoundHandler(QObject):
    alert_signal = pyqtSignal(str)  # Sinal para mensagens de alerta

    def __init__(self, alert_layout, alert_panel, results_label, parent=None):
        super().__init__(parent)
        self.alert_layout = alert_layout
        self.alert_panel = alert_panel
        self.results_label = results_label

        # Configuração única de som
        self.sound = None
        self.last_play_time = 0
        self.sound_cooldown = 1.5  # Tempo mínimo entre sons (1.5 segundos)

        # Timer para evitar sobreposição
        self.cooldown_timer = QTimer()
        self.cooldown_timer.setSingleShot(True)
        self.cooldown_timer.timeout.connect(self._reset_cooldown)

        self._load_sound()
        self.alert_signal.connect(self._handle_alert)

    def _load_sound(self):
        """Carrega o arquivo de som único"""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        sound_path = os.path.join(base_dir, "assets", "sounds", "defect_alert.wav")

        if os.path.exists(sound_path):
            self.sound = QSound(sound_path)
            logger.info(f"Som de defeito carregado com sucesso: {sound_path}")
        else:
            logger.warning(f"Arquivo de som não encontrado: {sound_path}")
            self.sound = None

    def _handle_alert(self, message):
        """Processa um alerta recebido"""
        current_time = time.time()

        # Verifica se pode tocar o som (cooldown)
        if current_time - self.last_play_time >= self.sound_cooldown:
            self._play_sound()
            self._show_alert(message)
            self.last_play_time = current_time
            self.cooldown_timer.start(int(self.sound_cooldown * 1000))
        else:
            # Se ainda está no cooldown, só mostra a mensagem
            self._show_alert(message)

    def _play_sound(self):
        """Toca o som de alerta"""
        if self.sound:
            self.sound.play()
        else:
            logger.warning("Som não carregado, alerta silencioso")

    def _show_alert(self, message):
        """Mostra o alerta visual"""
        alert = QLabel(message)
        alert.setWordWrap(True)
        alert.setMargin(8)
        alert.setStyleSheet(
            "background-color: #F8D7DA; "
            "border-left: 4px solid #DC3545; "
            "color: #721C24;"
        )
        self.alert_layout.insertWidget(0, alert)
        self.alert_panel.ensureWidgetVisible(alert)

    def _reset_cooldown(self):
        """Reseta o cooldown quando o timer expira"""
        pass  # Apenas para completar o ciclo do timer

    def trigger_alert(self, message, defect_key=None):
        """Dispara um alerta (interface pública)"""
        # No futuro você pode usar defect_key para evitar repetição de alertas
        self.alert_signal.emit(message)

    def cleanup(self):
        """Limpeza ao encerrar"""
        self.cooldown_timer.stop()
        if self.sound:
            self.sound.stop()
