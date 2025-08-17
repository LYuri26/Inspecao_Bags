import os
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QFileDialog,
    QLabel,
    QDateEdit,
    QTextEdit,
    QCompleter,
)
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtGui import QTextDocument
from PyQt5.QtCore import QDate, Qt
import json
import cv2
import numpy as np
import logging
import time
from .utils import save_defect_image

logger = logging.getLogger(__name__)


def process_detection(self, frame):
    """Roda a detecção no frame e retorna imagem anotada"""
    try:
        # roda inferência
        results = self.model.model(frame)[0]

        # se não houver detecção
        if not getattr(results, "boxes", None) or len(results.boxes) == 0:
            self._check_bag_timeout()
            return frame

        # cria cópia para anotações
        annotated = frame.copy()
        policy = self.active_company.get("policy", {}) if self.active_company else {}
        detected_bag = False

        for box, score, cls_id in zip(
            results.boxes.xyxy, results.boxes.conf, results.boxes.cls
        ):
            score = float(score)
            if score < 0.6:  # confiança mínima
                continue

            class_name = self.model.model.names[int(cls_id)]
            name_l = class_name.lower()
            name_d = class_name.capitalize()

            # coordenadas da bounding box
            x1, y1, x2, y2 = map(int, box.cpu().numpy())

            # atualiza controle de sacola
            if name_l == "sacola":
                detected_bag = True
                self.last_bag_seen_time = time.time()

            # checa política da empresa (se defeito é aceito, ignora)
            if name_l in self.defect_mapping and policy.get(
                self.defect_mapping[name_l], False
            ):
                continue

            # escolhe cor da classe
            color = self.class_colors.get(name_d, (255, 255, 255))

            # desenha bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{name_d} ({score:.2f})"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

            # trata defeito detectado
            if name_l in self.defect_mapping and not policy.get(
                self.defect_mapping[name_l], False
            ):
                self._handle_defect_detection(
                    camera_id=getattr(self, "current_camera", 0),
                    defect_name=name_d,
                    bbox=(x1, y1, x2, y2),
                    frame=annotated,
                )

        # timeout de troca de sacola
        if detected_bag and (time.time() - self.last_bag_seen_time > 15):
            self.bag_counter += 1
            self.current_bag_defects.clear()
            self.last_bag_seen_time = time.time()

        return annotated

    except Exception as e:
        logger.error(
            f"[process_detection] Erro no processamento: {str(e)}", exc_info=True
        )
        return frame


def display_image(self, image: np.ndarray):
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtCore import Qt

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    q_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)

    pixmap = QPixmap.fromImage(q_img)

    # Define o tamanho real da imagem no QLabel
    self.camera_label.setPixmap(
        pixmap.scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
    )


def generate_defect_summary(self, company_name, date_from, date_to):
    safe_name = "".join(
        c for c in company_name if c.isalnum() or c in (" ", "-", "_")
    ).rstrip()

    base_dir = Path(__file__).resolve()
    while base_dir.name != "Inspecao_Bags":
        base_dir = base_dir.parent

    cadastros_dir = base_dir / "cadastros"
    reports_dir = cadastros_dir / safe_name / "reports"
    documents_dir = cadastros_dir / safe_name / "documents"

    defect_summary = {}

    if not reports_dir.exists():
        return defect_summary

    for day_folder in reports_dir.iterdir():
        if not day_folder.is_dir():
            continue
        try:
            folder_date = datetime.strptime(day_folder.name, "%d-%m-%Y").date()
        except ValueError:
            continue
        if date_from.toPyDate() <= folder_date <= date_to.toPyDate():
            date_str = folder_date.strftime("%d/%m/%Y")
            defect_summary.setdefault(date_str, {})
            for img_file in day_folder.glob("*.jpg"):
                parts = img_file.name.replace(".jpg", "").split("-")
                bag_id = (
                    parts[1]
                    if len(parts) > 2 and parts[1].startswith("bag")
                    else "bag?"
                )
                defect_type = parts[-1].lower()
                defect_summary[date_str].setdefault(bag_id, {})
                defect_summary[date_str][bag_id][defect_type] = (
                    defect_summary[date_str][bag_id].get(defect_type, 0) + 1
                )

        # Salva documentos
        if not documents_dir.exists():
            documents_dir.mkdir()
        day_folder_name = date_to.toString("dd-MM-yyyy")
        day_report_dir = documents_dir / day_folder_name
        day_report_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = day_report_dir / f"summary_{timestamp}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Relatório de Inspeções - {company_name}\n")
            f.write(
                f"Período: {date_from.toString('dd/MM/yyyy')} a {date_to.toString('dd/MM/yyyy')}\n\n"
            )
            for date_str, bags in sorted(defect_summary.items(), reverse=True):
                f.write(f"Data: {date_str}\n")
                for bag, defects in bags.items():
                    f.write(f"  {bag}:\n")
                    for defect, count in defects.items():
                        f.write(f"    - {defect.capitalize()}: {count}\n")
                f.write("\n")

        return defect_summary
