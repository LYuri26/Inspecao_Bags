import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_main_window(self):
    parent = self.parent()
    while parent:
        if hasattr(parent, "views") and isinstance(parent.views, dict):
            return parent
        parent = parent.parent()
    return None


def update_status(self, message, error=False):
    color = "red" if error else "green"
    self.status_label.setText(f"Status: {message}")
    self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")


def set_active_company(self, company):
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
    """
    Salva a imagem do defeito detectado dentro da pasta da empresa selecionada,
    organizada por data, e nomeando com timestamp, id da câmera e nome do defeito.
    Também gera um TXT com feedback do evento.
    """
    if not self.active_company:
        return None

    try:
        company_name = self.active_company["name"]
        safe_name = "".join(
            c for c in company_name if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()

        # Localiza a pasta cadastros na raiz do projeto
        base_dir = Path(__file__).resolve()
        while base_dir.name != "Inspecao_Bags":
            base_dir = base_dir.parent
        cadastros_dir = base_dir / "cadastros"

        # Pasta de relatórios da empresa
        reports_dir = cadastros_dir / safe_name / "reports"

        # Pasta específica do dia
        day_folder_name = datetime.now().strftime("%d-%m-%Y")
        day_folder = reports_dir / day_folder_name
        day_folder.mkdir(parents=True, exist_ok=True)

        # Nome do arquivo
        timestamp = datetime.now().strftime("%H-%M-%S")
        camera_suffix = f"-cam{camera_id}" if camera_id is not None else ""
        filename = f"{timestamp}-{defect_type}{camera_suffix}.jpg"
        filepath = day_folder / filename

        # Salva imagem
        cv2.imwrite(str(filepath), frame)
        logger.info(f"Imagem de defeito salva em: {filepath}")

        # Cria/atualiza arquivo de log
        log_file = day_folder / "defects_log.txt"
        with open(log_file, "a", encoding="utf-8") as f:
            log_entry = (
                f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | "
                f"Empresa: {company_name} | "
                f"Câmera: {camera_id+1 if camera_id is not None else 'N/A'} | "
                f"Defeito: {defect_type} | "
                f"Arquivo: {filename}\n"
            )
            f.write(log_entry)

        return str(filepath)

    except Exception as e:
        logger.error(f"Erro ao salvar imagem do defeito: {str(e)}", exc_info=True)
        return None
