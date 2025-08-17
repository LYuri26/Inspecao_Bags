# src/ui/views/camera_module/camera_manager.py
from PyQt5.QtCore import QObject, pyqtSignal, QThread
import subprocess
import numpy as np
import logging
import time
import threading
from typing import Optional, List, Dict


logger = logging.getLogger(__name__)


class FFmpegWorker(QThread):
    """
    Worker que consome um RTSP com FFmpeg e emite frames (BGR) prontos pro OpenCV/Qt.
    Prioriza estabilidade e baixo consumo.
    """

    frame_captured = pyqtSignal(int, object)  # (camera_id, frame ndarray)

    def __init__(
        self,
        camera_id: int,
        rtsp_url: str,
        width: int = 320,
        height: int = 240,
        fps: int = 5,
        reconnect_delay: float = 2.0,
    ):
        super().__init__()
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.fps = fps
        self.reconnect_delay = reconnect_delay

        self._running = False
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None

        # Tamanho esperado de cada frame bgr24 (rawvideo)
        self._frame_size = self.width * self.height * 3

    # ---------- Processo FFmpeg ----------

    def _build_cmd(self) -> List[str]:
        # Comando simples e robusto — o mesmo padrão do seu script que funciona
        return [
            "ffmpeg",
            "-rtsp_transport",
            "tcp",
            "-i",
            self.rtsp_url,
            "-loglevel",
            "error",  # erros no stderr (a thread os consome)
            "-an",  # sem áudio
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-",  # stdout
        ]

    def _start_ffmpeg(self) -> bool:
        try:
            self._proc = subprocess.Popen(
                self._build_cmd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8,
            )
        except Exception as e:
            logger.error(f"❌ [Cam {self.camera_id}] Falha ao iniciar FFmpeg: {e}")
            self._proc = None
            return False

        logger.info(
            f"✅ FFmpeg iniciado para câmera {self.camera_id} ({self.rtsp_url})"
        )

        # Thread para drenar/logar stderr (evita travar o processo por buffer cheio)
        def _drain_stderr(proc: subprocess.Popen, cam_id: int):
            try:
                for line in iter(proc.stderr.readline, b""):
                    if not line:
                        break
                    msg = line.decode(errors="ignore").strip()
                    if msg:
                        logger.warning(f"[Cam {cam_id}] FFmpeg: {msg}")
            except Exception:
                pass

        self._stderr_thread = threading.Thread(
            target=_drain_stderr, args=(self._proc, self.camera_id), daemon=True
        )
        self._stderr_thread.start()
        return True

    def _stop_ffmpeg(self):
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None

    # ---------- Loop principal do worker ----------

    def run(self):
        self._running = True

        while self._running:
            # (Re)inicia FFmpeg se necessário
            if self._proc is None:
                if not self._start_ffmpeg():
                    # Falhou ao iniciar; espera e tenta de novo
                    time.sleep(self.reconnect_delay)
                    continue

            assert self._proc is not None
            stdout = self._proc.stdout

            if stdout is None:
                # Algo bem errado — reinicia
                logger.error(
                    f"⚠️ [Cam {self.camera_id}] stdout do FFmpeg ausente, reiniciando…"
                )
                self._stop_ffmpeg()
                time.sleep(self.reconnect_delay)
                continue

            # Lê 1 frame bgr24
            raw = stdout.read(self._frame_size)

            # Se não veio frame completo…
            if not raw or len(raw) < self._frame_size:
                # Se o processo morreu, reinicia. Senão, dá um respiro.
                if self._proc.poll() is not None:
                    logger.warning(
                        f"⚠️ [Cam {self.camera_id}] FFmpeg encerrou, tentando reconectar…"
                    )
                    self._stop_ffmpeg()
                    time.sleep(self.reconnect_delay)
                else:
                    time.sleep(0.01)
                continue

            # Converte para ndarray e emite
            try:
                frame = np.frombuffer(raw, np.uint8).reshape(
                    (self.height, self.width, 3)
                )
                self.frame_captured.emit(self.camera_id, frame)
            except Exception as e:
                logger.error(f"⚠️ [Cam {self.camera_id}] Erro ao montar frame: {e}")
                time.sleep(0.005)

        # Encerramento limpo
        self._stop_ffmpeg()

    def stop(self):
        self._running = False
        self.wait()


class CameraManager(QObject):
    """
    Gerencia até 9 câmeras. Mantém somente o frame mais recente por câmera.
    Compatível com CameraView: expõe `frame_ready` e `get_latest_frame`.
    """

    frame_ready = pyqtSignal(int, object)  # (camera_id, frame)

    def __init__(self, num_cameras: int = 9):
        super().__init__()
        self.num_cameras = num_cameras
        self.running = False

        # Configuração das URLs RTSP para todas as 9 câmeras
        self.camera_urls: List[Optional[str]] = [
            "rtsp://admin:Solutions10@@192.168.0.241:554/Streaming/Channels/101",  # Cam 0 "rtsp://admin:Evento0128@192.168.1.101:559/Streaming/Channels/101",  # Cam 0
            None,  # "rtsp://admin:Solutions10@@192.168.0.242:554/Streaming/Channels/101",  # Cam 1
            None,  # "rtsp://admin:Solutions10@@192.168.0.243:554/Streaming/Channels/101",  # Cam 2
            None,  # "rtsp://admin:Solutions10@@192.168.0.244:554/Streaming/Channels/101",  # Cam 3
            None,  # "rtsp://admin:Solutions10@@192.168.0.245:554/Streaming/Channels/101",  # Cam 4
            None,  # "rtsp://admin:Solutions10@@192.168.0.246:554/Streaming/Channels/101",  # Cam 5
            None,  # "rtsp://admin:Solutions10@@192.168.0.247:554/Streaming/Channels/101",  # Cam 6
            None,  # "rtsp://admin:Solutions10@@192.168.0.248:554/Streaming/Channels/101",  # Cam 7
            None,  # Cam 8 (reservada para futura expansão)
        ]
        # Parâmetros leves por padrão (pode ajustar depois)
        self.WIDTH = 320
        self.HEIGHT = 240
        self.FPS = 5

        # Últimos frames (um por câmera)
        self._latest_frames: Dict[int, np.ndarray] = {}

        # Workers
        self._workers: List[FFmpegWorker] = []

    # ---------- API pública ----------

    def set_camera_url(self, camera_id: int, rtsp_url: Optional[str]):
        """Permite configurar/atualizar a URL de uma câmera."""
        if 0 <= camera_id < self.num_cameras:
            self.camera_urls[camera_id] = rtsp_url

    def start_capture(self):
        if self.running:
            return
        self.running = True

        self._workers.clear()
        for cam_id, url in enumerate(self.camera_urls):
            if not url:
                continue

            worker = FFmpegWorker(
                camera_id=cam_id,
                rtsp_url=url,
                width=self.WIDTH,
                height=self.HEIGHT,
                fps=self.FPS,
            )
            worker.frame_captured.connect(self._on_frame)
            worker.start()
            self._workers.append(worker)

        if not self._workers:
            logger.warning("⚠️ Nenhuma URL de câmera configurada. Nada a iniciar.")

    def stop_capture(self):
        if not self.running:
            return
        self.running = False

        for w in self._workers:
            w.stop()
        self._workers.clear()

        self._latest_frames.clear()

    def get_latest_frame(self, camera_id: int):
        """Usado pelo CameraView para obter o frame atual."""
        return self._latest_frames.get(camera_id)

    # ---------- Internos ----------

    def _on_frame(self, cam_id: int, frame: np.ndarray):
        # Mantém somente o último frame (evita uso de memória crescente)
        self._latest_frames[cam_id] = frame
        # Emite para o CameraView redesenhar
        self.frame_ready.emit(cam_id, frame)
