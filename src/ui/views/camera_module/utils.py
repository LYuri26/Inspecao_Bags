import logging
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)


logger = logging.getLogger(__name__)


def get_main_window(self):
    parent = self.parent()
    while parent:
        if hasattr(parent, "views") and isinstance(parent.views, dict):
            return parent
        parent = parent.parent()
    return None


def update_status(self, message, error=False):
    """Atualiza a barra de status da interface"""
    color = "red" if error else "green"
    self.status_label.setText(f"Status: {message}")
    self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")


def set_active_company(self, company):
    """Define a empresa ativa e atualiza a interface"""
    self.active_company = company
    update_status(
        self,
        (
            f"Empresa selecionada: {company['name']}"
            if company
            else "Nenhuma empresa selecionada"
        ),
    )


def save_defect_image(self, frame: np.ndarray, defect_type: str, camera_id: int = None):
    if not self.active_company:
        logger.error("Nenhuma empresa ativa selecionada. Abortando save_defect_image.")
        return None

    try:
        company_name = self.active_company["name"]
        safe_name = "".join(
            c for c in company_name if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()

        logger.debug(
            f"Tentando salvar defeito para empresa: {company_name} ({safe_name})"
        )

        # Verifica base_path
        if not hasattr(self, "base_path"):
            logger.error("Objeto não possui atributo base_path")
            return None

        logger.debug(f"Base path definido em: {self.base_path}")

        reports_dir = Path(self.base_path) / safe_name / "reports"

        # Pasta específica do dia
        day_folder = reports_dir / datetime.now().strftime("%d-%m-%Y")
        logger.debug(f"Criando pasta de destino: {day_folder}")
        day_folder.mkdir(parents=True, exist_ok=True)

        # Nome do arquivo
        timestamp = datetime.now().strftime("%H-%M-%S")
        bag_suffix = f"-bag{self.bag_counter}" if hasattr(self, "bag_counter") else ""
        cam_suffix = f"-cam{camera_id+1}" if camera_id is not None else ""
        filename = f"{timestamp}{bag_suffix}-{defect_type}{cam_suffix}.jpg"
        filepath = day_folder / filename

        logger.debug(f"Caminho final da imagem: {filepath}")

        # Tenta salvar imagem
        success = cv2.imwrite(str(filepath), frame)
        if not success:
            logger.error(f"Falha ao salvar imagem com cv2.imwrite: {filepath}")
            return None

        logger.info(f"Imagem de defeito salva em: {filepath}")

        # Cria/atualiza log
        log_file = day_folder / "defects_log.txt"
        with open(log_file, "a", encoding="utf-8") as f:
            log_entry = (
                f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | "
                f"Empresa: {company_name} | "
                f"Câmera: {camera_id+1 if camera_id is not None else 'N/A'} | "
                f"Sacola: {getattr(self, 'bag_counter', 'N/A')} | "
                f"Defeito: {defect_type} | "
                f"Arquivo: {filename}\n"
            )
            f.write(log_entry)

        return str(filepath)

    except Exception as e:
        logger.error(f"Erro inesperado ao salvar defeito: {str(e)}", exc_info=True)
        return None


def display_image(self, image: np.ndarray):
    """Exibe imagem na interface"""
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtCore import Qt

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    q_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(q_img)
    self.camera_label.setPixmap(
        pixmap.scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
    )
