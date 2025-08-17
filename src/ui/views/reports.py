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


class ReportsView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.html_report = ""
        self.selected_company = None
        self.setup_ui()
        self.all_companies = self.get_all_companies()
        self.load_companies()

    def setup_ui(self):
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        filter_layout = QHBoxLayout()

        # ComboBox editável com filtro integrado
        self.company_combo = QComboBox()
        self.company_combo.setEditable(True)
        self.company_combo.setInsertPolicy(QComboBox.NoInsert)
        self.company_combo.completer().setCompletionMode(QCompleter.PopupCompletion)
        self.company_combo.lineEdit().setPlaceholderText("Pesquisar empresa...")
        self.company_combo.lineEdit().setClearButtonEnabled(True)
        self.company_combo.setStyleSheet(
            """
            QComboBox {
                padding: 8px;
                font-size: 14px;
                min-width: 300px;
            }
            QComboBox::drop-down {
                width: 20px;
            }
        """
        )

        self.date_from = QDateEdit()
        self.date_from.setDate(QDate.currentDate().addDays(-7))
        self.date_from.setCalendarPopup(True)
        self.date_from.setDisplayFormat("dd/MM/yyyy")

        self.date_to = QDateEdit()
        self.date_to.setDate(QDate.currentDate())
        self.date_to.setCalendarPopup(True)
        self.date_to.setDisplayFormat("dd/MM/yyyy")

        self.generate_btn = QPushButton("Gerar Relatório")
        self.generate_btn.setEnabled(False)
        self.generate_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2E8B57;
                color: white;
                padding: 8px 15px;
                border-radius: 4px;
                min-width: 120px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """
        )

        self.export_btn = QPushButton("Exportar PDF")
        self.export_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #4682B4;
                color: white;
                padding: 8px 15px;
                border-radius: 4px;
                min-width: 120px;
            }
        """
        )
        self.export_btn.setEnabled(False)

        filter_layout.addWidget(QLabel("Empresa:"))
        filter_layout.addWidget(self.company_combo, stretch=1)
        filter_layout.addWidget(QLabel("De:"))
        filter_layout.addWidget(self.date_from)
        filter_layout.addWidget(QLabel("Até:"))
        filter_layout.addWidget(self.date_to)
        filter_layout.addWidget(self.generate_btn)
        filter_layout.addWidget(self.export_btn)
        filter_layout.setSpacing(10)

        self.layout.addLayout(filter_layout, stretch=0)

        self.report_display = QTextEdit()
        self.report_display.setReadOnly(True)
        self.report_display.setStyleSheet(
            """
            QTextEdit {
                font-family: Arial;
                font-size: 12pt;
                border: 1px solid #ddd;
                padding: 15px;
                margin-top: 10px;
            }
        """
        )
        self.layout.addWidget(self.report_display, stretch=1)

        # Conexões
        self.company_combo.currentTextChanged.connect(self.on_company_changed)
        self.generate_btn.clicked.connect(self.generate_report)
        self.export_btn.clicked.connect(self.export_pdf)

    def get_all_companies(self):
        base_dir = Path(__file__).resolve()
        while base_dir.name != "Inspecao_Sacolas":
            base_dir = base_dir.parent
        cadastros_dir = base_dir / "cadastros"

        companies = []
        if not cadastros_dir.exists():
            return companies

        for company_folder in cadastros_dir.iterdir():
            if not company_folder.is_dir():
                continue
            json_file = company_folder / f"{company_folder.name}.json"
            if json_file.exists():
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        company_data = json.load(f)
                        companies.append(company_data.get("name", company_folder.name))
                except Exception as e:
                    print(f"Erro ao carregar {json_file}: {e}")

        return sorted(companies, key=lambda x: x.lower())

    def load_companies(self):
        self.company_combo.clear()
        self.company_combo.addItem("")  # item vazio inicial
        for company in self.all_companies:
            self.company_combo.addItem(company)
        self.company_combo.completer().setFilterMode(Qt.MatchContains)
        self.company_combo.completer().setCaseSensitivity(Qt.CaseInsensitive)

    def on_company_changed(self, text):
        text = text.strip()
        if text and text in self.all_companies:
            self.generate_btn.setEnabled(True)
            self.selected_company = text
        else:
            self.generate_btn.setEnabled(False)
            self.selected_company = None

    def generate_report(self):
        if not self.selected_company:
            return

        company = self.selected_company
        date_from = self.date_from.date()
        date_to = self.date_to.date()

        defect_summary = self.generate_defect_summary(company, date_from, date_to)

        # Cálculos KPIs
        total_bags = 0
        total_defects = 0
        defects_by_type = {}
        defects_by_date = {}

        for date_str, bags in defect_summary.items():
            defects_count_date = 0
            for bag_id, defects in bags.items():
                total_bags += 1
                for defect, count in defects.items():
                    defects_by_type[defect] = defects_by_type.get(defect, 0) + count
                    defects_count_date += count
            defects_by_date[date_str] = defects_count_date
            total_defects += defects_count_date

        # Evitar divisão por zero
        defect_rate = (total_defects / total_bags * 100) if total_bags > 0 else 0

        # Defeito mais comum
        top_defect = (
            max(defects_by_type.items(), key=lambda x: x[1])[0]
            if defects_by_type
            else "Nenhum"
        )

        # Datas com mais defeitos (top 3)
        dates_sorted = sorted(defects_by_date.items(), key=lambda x: x[1], reverse=True)
        dates_with_most_defects = (
            ", ".join([d[0] for d in dates_sorted[:3]]) if dates_sorted else "Nenhuma"
        )

        # Construir listas HTML de defeitos por tipo
        defects_by_type_html = "".join(
            f"<li>{defect.capitalize()}: {count}</li>"
            for defect, count in defects_by_type.items()
        )

        # Construir lista HTML de defeitos por data
        defects_by_date_html = "".join(
            f"<li>{date}: {count} defeito(s)</li>" for date, count in dates_sorted
        )

        # HTML do relatório completo
        self.html_report = f"""
        <html>
        <head><meta charset="UTF-8"><title>Relatório Completo de Inspeções</title></head>
        <body style="font-family: Arial; margin: 20px;">
            <h1 style="color: #2E8B57; border-bottom: 1px solid #ddd; padding-bottom: 10px;">
                Relatório Completo de Inspeções
            </h1>
            <div style="margin-bottom: 20px;">
                <p><b>Empresa:</b> {company}</p>
                <p><b>Período:</b> {date_from.toString("dd/MM/yyyy")} a {date_to.toString("dd/MM/yyyy")}</p>
                <p><b>Data de emissão:</b> {datetime.now().strftime("%d/%m/%Y %H:%M")}</p>
            </div>

            <h2>Resumo Executivo</h2>
            <p>
                Durante o período de <b>{date_from.toString("dd/MM/yyyy")} a {date_to.toString("dd/MM/yyyy")}</b>, a inspeção da empresa <b>{company}</b> analisou 
                <b>{total_bags}</b> sacolas, identificando <b>{total_defects}</b> defeitos, com uma taxa média de <b>{defect_rate:.2f}%</b>. 
                O defeito mais comum foi <b>{top_defect.capitalize()}</b>, indicando possível falha no processo de produção.
            </p>

            <h2>Indicadores-Chave</h2>
            <ul>
                <li>Total de Sacolas Inspecionadas: {total_bags}</li>
                <li>Total de Defeitos Identificados: {total_defects}</li>
                <li>Defeitos por Tipo:</li>
                <ul>
                    {defects_by_type_html}
                </ul>
                <li>Distribuição Diária de Defeitos:</li>
                <ul>
                    {defects_by_date_html}
                </ul>
            </ul>

            <h2>Análise Detalhada</h2>
            <p>
                Observou-se que as datas <b>{dates_with_most_defects}</b> apresentaram picos significativos de defeitos,
                sugerindo necessidade de avaliação do processo produtivo nestes dias.
            </p>

            <h2>Recomendações</h2>
            <ul>
                <li>Revisar configuração da máquina para reduzir ocorrência de <b>{top_defect.capitalize()}</b>.</li>
                <li>Aumentar a frequência de inspeção manual nos períodos críticos identificados.</li>
            </ul>

            <h2>Registros Visuais</h2>
        """

        # Acrescenta as imagens existentes ao relatório (pegando método que você já tem)
        images_html = ""
        images = self.get_defect_images(company, date_from, date_to)
        if images:
            images_html = "<div style='display: flex; flex-wrap: wrap; gap: 15px;'>"
            for img in images:
                images_html += f"""
                <div style='width: 200px; border: 1px solid #ddd; padding: 5px;'>
                    <img src='file://{img['path']}' style='width: 100%; height: auto;' />
                    <div style='text-align: center; font-size: 10pt; margin-top: 5px;'>
                        {img['date']} - {img['bag'].upper()} - {img['defect'].capitalize()}
                    </div>
                </div>
                """
            images_html += "</div>"
        else:
            images_html = (
                "<p>Nenhum registro visual disponível para o período selecionado.</p>"
            )

        self.html_report += images_html

        self.html_report += """
            <h2>Conclusão</h2>
            <p>
                O monitoramento constante e as ações corretivas recomendadas são fundamentais para garantir a qualidade e
                a satisfação dos clientes. A inspeção automatizada demonstrou ser uma ferramenta valiosa neste processo.
            </p>
        </body>
        </html>
        """

        self.report_display.setHtml(self.html_report)
        self.export_btn.setEnabled(True)

    def export_pdf(self):
        if not self.html_report:
            return

        default_name = (
            f"relatorio_{self.selected_company}_{datetime.now().strftime('%Y%m%d')}.pdf"
        )
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Exportar PDF", default_name, "PDF Files (*.pdf)"
        )

        if file_name:
            if not file_name.endswith(".pdf"):
                file_name += ".pdf"

            doc = QTextDocument()
            doc.setHtml(self.html_report)

            printer = QPrinter(QPrinter.HighResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(file_name)
            printer.setPageMargins(15, 15, 15, 15, QPrinter.Millimeter)

            doc.print_(printer)

    def get_defect_images(self, company_name, date_from, date_to):
        safe_name = "".join(
            c for c in company_name if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()

        base_dir = Path(__file__).resolve()
        while base_dir.name != "Inspecao_Sacolas":
            base_dir = base_dir.parent
        cadastros_dir = base_dir / "cadastros"
        reports_dir = cadastros_dir / safe_name / "reports"

        if not reports_dir.exists():
            return []

        images = []
        for day_folder in reports_dir.iterdir():
            if not day_folder.is_dir():
                continue
            try:
                folder_date = datetime.strptime(day_folder.name, "%d-%m-%Y").date()
            except ValueError:
                continue
            if date_from.toPyDate() <= folder_date <= date_to.toPyDate():
                for img_file in day_folder.glob("*.jpg"):
                    # Extrair partes do nome do arquivo
                    parts = img_file.name.replace(".jpg", "").split("-")
                    bag_id = next((p for p in parts if p.startswith("bag")), "bag?")
                    defect_name = parts[-1] if parts else "desconhecido"

                    images.append(
                        {
                            "path": str(img_file.resolve()),
                            "date": folder_date.strftime("%d/%m/%Y"),
                            "bag": bag_id,
                            "defect": defect_name,
                        }
                    )

        return sorted(
            images, key=lambda x: datetime.strptime(x["date"], "%d/%m/%Y"), reverse=True
        )

    def _generate_html(self, company, date_from, date_to, defect_summary):
        summary_html = "<div style='margin-bottom: 30px;'>"
        summary_html += "<h2 style='color: #2E8B57;'>Resumo de Defeitos</h2>"

        if not defect_summary:
            summary_html += "<p>Nenhum defeito encontrado no período selecionado.</p>"
        else:
            for date_str, bags in sorted(defect_summary.items(), reverse=True):
                summary_html += f"<h3 style='margin-top: 15px;'>{date_str}</h3><ul>"
                for bag_id, defects in bags.items():
                    summary_html += f"<li><b>{bag_id.upper()}</b><ul>"
                    for defect, count in defects.items():
                        summary_html += f"<li>{defect.capitalize()}: {count}</li>"
                    summary_html += "</ul></li>"
                summary_html += "</ul>"
        summary_html += "</div>"

        # Carregar imagens já com identificação de bag
        images = self.get_defect_images(company, date_from, date_to)
        images_html = ""
        if images:
            images_html = "<div style='margin-top: 30px;'>"
            images_html += "<h2 style='color: #2E8B57;'>Registros Visuais</h2>"
            images_html += "<div style='display: flex; flex-wrap: wrap; gap: 15px;'>"

            for img in images:
                parts = os.path.basename(img["path"]).replace(".jpg", "").split("-")
                bag_id = next((p for p in parts if p.startswith("bag")), "bag?")
                defect_name = img.get("defect", parts[-1])
                images_html += f"""
                <div style='width: 200px; border: 1px solid #ddd; padding: 5px;'>
                    <img src='file://{img['path']}' style='width: 100%; height: auto;' />
            <div style='text-align: center; font-size: 10pt; margin-top: 5px;'>
                {img['date']} - {img['bag'].upper()} - {img['defect'].capitalize()}
            </div>
                </div>
                """
            images_html += "</div></div>"

        return f"""
        <html>
        <head><meta charset="UTF-8"><title>Relatório de Inspeções</title></head>
        <body style="font-family: Arial; margin: 20px;">
            <h1 style="color: #2E8B57; border-bottom: 1px solid #ddd; padding-bottom: 10px;">
                Relatório de Inspeções
            </h1>
            <div style="margin-bottom: 20px;">
                <p><b>Empresa:</b> {company}</p>
                <p><b>Período:</b> {date_from.toString("dd/MM/yyyy")} a {date_to.toString("dd/MM/yyyy")}</p>
                <p><b>Data de emissão:</b> {datetime.now().strftime("%d/%m/%Y %H:%M")}</p>
            </div>
            {summary_html}
            {images_html}
        </body>
        </html>
        """

    def generate_defect_summary(self, company_name, date_from, date_to):
        safe_name = "".join(
            c for c in company_name if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()

        base_dir = Path(__file__).resolve()
        while base_dir.name != "Inspecao_Sacolas":
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
                    bag_id = next((p for p in parts if p.startswith("bag")), "bag?")
                    defect_type = parts[-1].lower()

                    defect_summary[date_str].setdefault(bag_id, {})
                    defect_summary[date_str][bag_id][defect_type] = (
                        defect_summary[date_str][bag_id].get(defect_type, 0) + 1
                    )

        # Salva documentos do relatório
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
                for bag_id, defects in bags.items():
                    f.write(f"  {bag_id.upper()}:\n")
                    for defect, count in defects.items():
                        f.write(f"    - {defect.capitalize()}: {count}\n")
                f.write("\n")

        json_path = day_report_dir / f"summary_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(
                {
                    "company": company_name,
                    "date_from": date_from.toString("yyyy-MM-dd"),
                    "date_to": date_to.toString("yyyy-MM-dd"),
                    "generated_at": datetime.now().isoformat(),
                    "defects": defect_summary,
                },
                jf,
                indent=4,
                ensure_ascii=False,
            )

        return defect_summary
