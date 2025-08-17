import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np


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


def save_defect_image(self, frame: np.ndarray, defect_type: str):
    if not self.active_company:
        return

    company_name = self.active_company["name"]
    safe_name = "".join(
        c for c in company_name if c.isalnum() or c in (" ", "-", "_")
    ).rstrip()

    # Localiza a pasta cadastros na raiz do projeto
    base_dir = Path(__file__).resolve()
    while base_dir.name != "Inspecao_Sacolas":
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
    filename = f"{timestamp}-{safe_name}-{defect_type}.jpg"
    filepath = day_folder / filename

    # Salva a imagem
    cv2.imwrite(str(filepath), frame)
    return str(filepath)
