import sys
from pathlib import Path
import numpy as np
import cv2
from datetime import datetime

# garante que a pasta src esteja no sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

from ui.views.camera_module.camera_view import CameraView
from ui.views.reports import ReportsView
from PyQt5.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # cria imagem fake com defeito marcado
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 255), 3)

    cam = CameraView()
    cam.active_company = {"name": "Lenon Yuri", "policy": {}}
    cam.bag_counter = 0

    # força salvar imagem
    cam._save_defect_image(img, "rasgo", camera_id=0)

    # agora gera resumo
    reports = ReportsView()
    defect_summary = reports.generate_defect_summary(
        "Lenon Yuri", reports.date_from.date(), reports.date_to.date()
    )
    print("Resumo de defeitos gerado:")
    print(defect_summary)
