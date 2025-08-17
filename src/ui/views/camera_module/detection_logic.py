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


def process_detection(self, frame: np.ndarray) -> np.ndarray:
    """
    Processa um frame para detecção de defeitos com:
    - Registro único por localização aproximada por bag
    - Sincronização de bag_id entre as 9 câmeras
    - Troca automática de bag após 15s sem detectar sacola
    """
    try:
        annotated_frame = frame.copy()

        # Verifica modelo
        if not self.model or not hasattr(self.model, "model"):
            cv2.putText(
                annotated_frame,
                "Modelo não carregado",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return annotated_frame

        # Inferência YOLO
        result = self.model.model(frame)[0]
        if not getattr(result, "boxes", None) or len(result.boxes) == 0:
            # Verifica timeout de troca de bag
            if time.time() - self.last_bag_seen_time > 15:
                self.bag_counter += 1
                self.current_bag_defects.clear()
                self.last_bag_seen_time = time.time()
            return annotated_frame

        policy = self.active_company.get("policy", {}) if self.active_company else {}
        class_counts = {}
        detected_bag = False

        for box, score, cls_id in zip(
            result.boxes.xyxy, result.boxes.conf, result.boxes.cls
        ):
            score = float(score)
            if score < 0.6:
                continue

            class_name = self.model.model.names[int(cls_id)]
            name_l = class_name.lower()
            name_d = class_name.capitalize()

            x1, y1, x2, y2 = map(int, box.cpu().numpy())

            # Se detectar sacola, atualiza tempo
            if name_l == "sacola":
                detected_bag = True
                self.last_bag_seen_time = time.time()

            # Defeito permitido pela política? pula
            if name_l in self.defect_mapping and policy.get(
                self.defect_mapping[name_l], False
            ):
                continue

            # Desenha bounding box
            color = self.class_colors.get(name_d, (255, 255, 255))
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            label = f"{name_d} #{class_counts[class_name]} ({score:.2f})"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                annotated_frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1
            )
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

            # Registro de defeito único por localização aproximada
            if name_l in self.defect_mapping and not policy.get(
                self.defect_mapping[name_l], False
            ):
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                pos_key = (name_l, cx // 20, cy // 20)  # agrupa por blocos de 20px
                if pos_key not in self.current_bag_defects:
                    self.current_bag_defects.append(pos_key)
                    self.sound_handler.trigger_alert(
                        f"Defeito detectado: {name_d} (Bag {self.bag_counter})",
                        defect_key=name_l,
                    )

                    # Salva frame anotado
                    save_defect_image(
                        self, annotated_frame, f"bag{self.bag_counter + 1}-{name_l}"
                    )

                    # 🔹 Salva frame original (sem desenho)
                    orig_frame = self.camera_manager.get_latest_frame(
                        self.camera_id, raw=True
                    )
                    if orig_frame is not None:
                        from pathlib import Path
                        import cv2, datetime

                        day_folder = (
                            Path("cadastros")
                            / "raw_frames"
                            / datetime.now().strftime("%d-%m-%Y")
                        )
                        day_folder.mkdir(parents=True, exist_ok=True)

                        filename = f"bag{self.bag_counter + 1}-{name_l}-raw.jpg"
                        cv2.imwrite(str(day_folder / filename), orig_frame)

        # Se detectou nova sacola mas passou timeout, troca
        if detected_bag and (time.time() - self.last_bag_seen_time > 5):
            self.bag_counter += 1
            self.current_bag_defects.clear()
            self.last_bag_seen_time = time.time()

        return annotated_frame

    except Exception as e:
        logger.error(f"[process_detection] Erro no processamento: {e}", exc_info=True)
        return frame


def display_image(self, image: np.ndarray):
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtCore import Qt

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    q_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)

    pixmap = QPixmap.fromImage(q_img)

    # Define o tamanho real da imagem no QLabel
    self.camera_label.setPixmap(pixmap)
    self.camera_label.resize(pixmap.size())


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
