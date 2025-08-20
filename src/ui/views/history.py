from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QHBoxLayout,
    QPushButton,
    QDateEdit,
    QLabel,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QApplication,
)
from PyQt5.QtCore import Qt, QDate
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtGui import QTextDocument
import json
from pathlib import Path
from datetime import datetime
import base64
import builtins


class HistoryView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.reports = []

        # captura resolução da tela para ajustar escala
        screen = QApplication.primaryScreen().size()
        self.scale_factor = screen.width() / 1920  # base FullHD

        self.setup_ui()

    def setup_ui(self):
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        # ---------------- Filtros ----------------
        filter_layout = QHBoxLayout()
        self.company_combo = QComboBox()
        self.company_combo.setStyleSheet(
            f"font-size: {int(12*self.scale_factor)}px; padding:4px;"
        )

        self.date_from = QDateEdit()
        self.date_from.setDate(QDate.currentDate().addDays(-7))
        self.date_to = QDateEdit()
        self.date_to.setDate(QDate.currentDate())

        self.load_btn = QPushButton("Carregar Relatórios")
        self.load_btn.setMinimumHeight(int(35 * self.scale_factor))
        self.load_btn.setStyleSheet(
            f"font-size: {int(12*self.scale_factor)}px; font-weight: bold; padding:6px;"
        )

        filter_layout.addWidget(QLabel("Empresa:"))
        filter_layout.addWidget(self.company_combo)
        filter_layout.addWidget(QLabel("De:"))
        filter_layout.addWidget(self.date_from)
        filter_layout.addWidget(QLabel("Até:"))
        filter_layout.addWidget(self.date_to)
        filter_layout.addWidget(self.load_btn)
        self.layout.addLayout(filter_layout)

        # ---------------- Tabela ----------------
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Data", "Arquivo", "Defeito", "Empresa", "Resumo", "Ações"]
        )

        header = self.table.horizontalHeader()
        for i in range(5):
            header.setSectionResizeMode(i, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # coluna de botões

        self.layout.addWidget(self.table)

        # ---------------- Conexões ----------------
        self.load_btn.clicked.connect(self.load_reports)

        # Inicializa empresas
        self.load_companies()

    # ---------------- Empresas ----------------
    def load_companies(self):
        """Carrega lista de empresas a partir da pasta cadastros (sem duplicação)."""
        cadastros_dir = builtins.BASE_DIR / "cadastros"
        if not cadastros_dir.exists():
            return

        added = set()
        for company_dir in cadastros_dir.iterdir():
            if company_dir.is_dir():
                safe_name = company_dir.name.strip()
                if safe_name not in added:
                    self.company_combo.addItem(safe_name)
                    added.add(safe_name)

    # ---------------- Relatórios ----------------
    def load_reports(self):
        """Carrega relatórios da empresa selecionada dentro do período."""
        company = self.company_combo.currentText()
        if not company:
            QMessageBox.warning(self, "Histórico", "Selecione uma empresa.")
            return

        docs_dir = builtins.BASE_DIR / "cadastros" / company / "documents"

        if not docs_dir.exists():
            QMessageBox.information(
                self, "Histórico", "Nenhum relatório encontrado para esta empresa."
            )
            return

        dfrom = self.date_from.date().toPyDate()
        dto = self.date_to.date().toPyDate()

        self.reports = []
        rows = []

        for day_folder in docs_dir.iterdir():
            if not day_folder.is_dir():
                continue

            # aceita YYYY-MM-DD ou DD-MM-YYYY
            folder_date = None
            for fmt in ("%Y-%m-%d", "%d-%m-%Y"):
                try:
                    folder_date = datetime.strptime(day_folder.name, fmt).date()
                    break
                except Exception:
                    continue
            if not folder_date or not (dfrom <= folder_date <= dto):
                continue

            for jf in day_folder.glob("summary_*.json"):
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.reports.append({"file": jf, "data": data})

                        total_bags = sum(
                            len(b) for b in data.get("defects", {}).values()
                        )
                        rows.append(
                            [
                                folder_date.strftime("%d/%m/%Y"),
                                jf.name,
                                "-",
                                company,
                                f"{total_bags} bags",
                            ]
                        )
                except Exception as e:
                    print(f"⚠️ Erro ao carregar {jf}: {e}")

        # Preencher tabela
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, j, item)

            # Botão "Gerar PDF" responsivo
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setAlignment(Qt.AlignCenter)

            btn = QPushButton("Gerar PDF")
            btn.setMinimumSize(
                int(120 * self.scale_factor), int(35 * self.scale_factor)
            )
            btn.setStyleSheet(
                f"font-size: {int(12*self.scale_factor)}px; font-weight: bold; padding:4px;"
            )
            btn.clicked.connect(lambda _, r=i: self.export_report_to_pdf(r))

            layout.addWidget(btn)
            self.table.setCellWidget(i, 5, container)

        QMessageBox.information(
            self, "Histórico", f"{len(rows)} relatórios carregados."
        )

    # ---------------- Exportação ----------------
    def export_report_to_pdf(self, row):
        """Exporta o relatório individual da linha selecionada"""
        report_info = self.reports[row]
        data = report_info["data"]
        html = self._generate_html_report(data)

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar Relatório para PDF",
            f"{report_info['file'].stem}.pdf",
            "PDF Files (*.pdf)",
        )
        if not file_name:
            return
        if not file_name.lower().endswith(".pdf"):
            file_name += ".pdf"

        doc = QTextDocument()
        doc.setHtml(html)
        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(file_name)
        doc.print_(printer)

        QMessageBox.information(self, "Exportar", f"Relatório exportado: {file_name}")

    def _generate_html_report(self, data):
        """Gera HTML completo do histórico com imagens e estatísticas"""
        company = data.get("company", "Desconhecida").strip()
        date = data.get("date", "")
        defects = data.get("defects", {})

        # Diretórios base
        cadastros_dir = builtins.BASE_DIR / "cadastros"
        company_dir = cadastros_dir / company
        reports_dir = company_dir / "reports"

        # ---------- KPIs ----------
        total_bags = 0
        total_defects = 0
        defects_by_type = {}
        defects_by_date = {}

        for date_str, bags in defects.items():
            defects_count_date = 0
            for bag_id, bag_defects in bags.items():
                total_bags += 1
                for defect, value in bag_defects.items():
                    # Corrige bug: value pode ser int ou dict com {"count": int}
                    count = value["count"] if isinstance(value, dict) else value
                    defects_by_type[defect] = defects_by_type.get(defect, 0) + count
                    defects_count_date += count
            defects_by_date[date_str] = defects_count_date
            total_defects += defects_count_date

        defect_rate = (total_defects / total_bags * 100) if total_bags > 0 else 0
        top_defect = (
            max(defects_by_type.items(), key=lambda x: x[1])[0]
            if defects_by_type
            else "Nenhum"
        )
        dates_sorted = sorted(defects_by_date.items(), key=lambda x: x[1], reverse=True)
        dates_with_most_defects = (
            ", ".join([d[0] for d in dates_sorted[:3]]) if dates_sorted else "Nenhuma"
        )

        # ---------- Buscar imagens ----------
        images_html = ""
        img_paths = []

        try:
            dt = datetime.strptime(date, "%d/%m/%Y")
            day_folder = reports_dir / dt.strftime("%d-%m-%Y")
        except Exception:
            day_folder = reports_dir / date.replace("/", "-")

        if day_folder.exists():
            img_paths = list(day_folder.glob("*.jpg")) + list(day_folder.glob("*.png"))

        if img_paths:
            for img_path in img_paths:
                try:
                    with open(img_path, "rb") as imgf:
                        b64 = base64.b64encode(imgf.read()).decode("utf-8")
                        images_html += f"""
                        <div style='width:200px; display:inline-block; margin:10px; text-align:center;'>
                            <img src='data:image/png;base64,{b64}' style='width:100%; height:auto; border:1px solid #ccc; padding:3px;'/>
                            <div style='font-size:10pt; margin-top:5px;'>{img_path.stem}</div>
                        </div>
                        """
                except Exception as e:
                    print(f"⚠️ Erro ao carregar imagem {img_path}: {e}")
            feedback_msg = f"✅ {len(img_paths)} imagens encontradas em {day_folder}"
        else:
            images_html = (
                "<p>⚠ Nenhum registro visual disponível para este relatório.</p>"
            )
            feedback_msg = f"⚠ Nenhuma imagem encontrada em {day_folder}"

        print(feedback_msg)
        QMessageBox.information(self, "Histórico", feedback_msg)

        # ---------- Montar HTML ----------
        defects_by_type_html = "".join(
            f"<li>{defect.capitalize()}: {count}</li>"
            for defect, count in defects_by_type.items()
        )
        defects_by_date_html = "".join(
            f"<li>{date}: {count} defeito(s)</li>" for date, count in dates_sorted
        )

        html = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Histórico - {company}</title>
        </head>
        <body style="font-family: Arial; margin: 20px;">
            <h1 style="color:#2E8B57;">Histórico de Relatórios</h1>
            <h2>Empresa: {company}</h2>
            <p><b>Data:</b> {date}</p>

            <h2>Resumo Executivo</h2>
            <ul>
                <li>Sacolas Inspecionadas: {total_bags}</li>
                <li>⚠ Defeitos Identificados: {total_defects}</li>
                <li>Taxa de Defeitos: {defect_rate:.2f}%</li>
                <li>Defeito Mais Frequente: {top_defect}</li>
                <li>Datas com Mais Defeitos: {dates_with_most_defects}</li>
            </ul>

            <h2>Distribuição de Defeitos</h2>
            <ul>{defects_by_type_html}</ul>

            <h2>Defeitos por Data</h2>
            <ul>{defects_by_date_html}</ul>

            <h2>Registros Visuais</h2>
            {images_html}

            <h2>Conclusão</h2>
            <p>O acompanhamento diário permite identificar falhas recorrentes e agir preventivamente.</p>
        </body>
        </html>
        """
        return html
