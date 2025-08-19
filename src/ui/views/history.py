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
        """Carrega lista de empresas a partir da pasta cadastros"""
        base_dir = Path(__file__).resolve()
        while base_dir.name != "Inspecao_Bags":
            base_dir = base_dir.parent
        cadastros_dir = base_dir / "cadastros"

        if not cadastros_dir.exists():
            return

        for company_dir in cadastros_dir.iterdir():
            if company_dir.is_dir():
                self.company_combo.addItem(company_dir.name)

    # ---------------- Relatórios ----------------
    def load_reports(self):
        """Carrega relatórios da empresa selecionada dentro do período."""
        company = self.company_combo.currentText()
        if not company:
            QMessageBox.warning(self, "Histórico", "Selecione uma empresa.")
            return

        base_dir = Path(__file__).resolve()
        while base_dir.name != "Inspecao_Bags":
            base_dir = base_dir.parent
        docs_dir = base_dir / "cadastros" / company / "documents"

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
        company = data.get("company", "Desconhecida")
        date_from = data.get("date_from", "")
        date_to = data.get("date_to", "")
        defects = data.get("defects", {})

        defects_html = ""
        for date_str, bags in defects.items():
            defects_html += f"<h3>{date_str}</h3><ul>"
            for bag, defect_types in bags.items():
                defects_html += f"<li><b>{bag.upper()}</b><ul>"
                for defect, info in defect_types.items():
                    # info pode ser apenas contagem ou dict com imagens
                    if isinstance(info, dict) and "count" in info:
                        count = info["count"]
                        defects_html += f"<li>{defect.capitalize()}: {count}</li>"

                        # se tiver imagens
                        for img_path in info.get("images", []):
                            try:
                                with open(img_path, "rb") as imgf:
                                    b64 = base64.b64encode(imgf.read()).decode("utf-8")
                                    defects_html += f"<br><img src='data:image/png;base64,{b64}' style='max-width:400px; margin:5px;'>"
                            except Exception as e:
                                print(f"⚠️ Erro ao carregar imagem {img_path}: {e}")
                defects_html += "</ul></li>"
            defects_html += "</ul>"

        return f"""
        <html>
        <head><meta charset="UTF-8"><title>Relatório</title></head>
        <body style="font-family: Arial; margin: 20px;">
            <h1 style="color: #2E8B57;">Relatório de Inspeções</h1>
            <p><b>Empresa:</b> {company}</p>
            <p><b>Período:</b> {date_from} a {date_to}</p>
            <p><b>Data de emissão:</b> {data.get("generated_at","")}</p>
            <h2>Resumo de Defeitos</h2>
            {defects_html}
        </body>
        </html>
        """
