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
from pathlib import Path
import json
import builtins


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
        cadastros_dir = builtins.BASE_DIR / "cadastros"
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
                        # 🔹 Sempre usar o nome do JSON se existir
                        company_name = company_data.get("name")
                        if company_name:
                            companies.append(company_name.strip())
                        else:
                            companies.append(company_folder.name.strip())
                except Exception as e:
                    print(f"Erro ao carregar {json_file}: {e}")
            else:
                companies.append(company_folder.name.strip())

        # 🔹 Remover duplicatas mantendo apenas nomes únicos
        companies = list(dict.fromkeys(companies))
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
        """
        Gera relatório completo (HTML e JSON) para a empresa selecionada.
        Inclui caminhos das imagens em cada defeito no JSON.
        Usa caminhos relativos para funcionar em outros computadores.
        """
        if not self.selected_company:
            print("⚠️ Nenhuma empresa ativa selecionada. Relatório não gerado.")
            return

        company = self.selected_company.strip()
        cadastros_dir = builtins.BASE_DIR / "cadastros"
        company_dir = cadastros_dir / company

        date_from = self.date_from.date()
        date_to = self.date_to.date()

        # 🔹 Carregar imagens detectadas no período
        images = self.get_defect_images(company, date_from, date_to)
        if not images:
            print(f"ℹ️ Nenhum defeito encontrado para {company} no período.")
            return

        # 🔹 Resumo de defeitos (data → bag → defeito → {count, images})
        defect_summary_data = {}
        for img in images:
            date_str = img["date"]
            bag_id = img["bag"]
            defect = img["defect"]
            img_path = Path(img["path"])

            bag_defects = defect_summary_data.setdefault(date_str, {}).setdefault(
                bag_id, {}
            )
            entry = bag_defects.setdefault(defect, {"count": 0, "images": []})

            entry["count"] += 1

            # Gera caminho relativo a partir da pasta da empresa
            try:
                relative_path = str(img_path.relative_to(company_dir))
            except ValueError:
                relative_path = str(img_path)  # fallback para absoluto se der erro

            if relative_path not in entry["images"]:
                entry["images"].append(relative_path)

        # 🔹 Consolida / atualiza relatório diário no JSON
        defect_summary = ReportsView.generate_defect_summary(
            company, defect_summary_data, date_from, date_to
        )

        # JSON de saída
        day_folder_name = date_to.toString("yyyy-MM-dd")
        json_file = (
            builtins.BASE_DIR
            / "cadastros"
            / company
            / "documents"
            / day_folder_name
            / f"summary_{day_folder_name}.json"
        )
        json_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(defect_summary, f, indent=4, ensure_ascii=False)
            print(f"✅ Relatório JSON consolidado: {json_file}")
        except Exception as e:
            print(f"❌ Erro ao salvar JSON: {e}")

        # 🔹 KPIs
        total_bags, total_defects, defects_by_type, defects_by_date = 0, 0, {}, {}
        for date_str, bags in defect_summary["defects"].items():
            defects_count_date = 0
            for bag_id, defects in bags.items():
                total_bags += 1
                for defect, values in defects.items():
                    count = values["count"]
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

        # 🔹 HTML (usando caminho relativo no <img>)
        defects_by_type_html = "".join(
            f"<li>{defect.capitalize()}: {count}</li>"
            for defect, count in defects_by_type.items()
        )
        defects_by_date_html = "".join(
            f"<li>{date}: {count} defeito(s)</li>" for date, count in dates_sorted
        )

        images_html = ""
        for date_str, bags in defect_summary["defects"].items():
            for bag_id, defects in bags.items():
                for defect, values in defects.items():
                    for img_rel in values.get("images", []):
                        img_path = company_dir / img_rel
                        images_html += f"""
                        <div style='width:200px; border:1px solid #ddd; margin:5px; padding:5px;'>
                            <img src='file://{img_path}' style='width:100%; height:auto;' />
                            <div style='text-align:center; font-size:10pt; margin-top:5px;'>
                                {date_str} - {bag_id.upper()} - {defect.capitalize()}
                            </div>
                        </div>
                        """

        self.html_report = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2E86C1; }}
                .section {{ margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>Relatório de Defeitos - {company}</h1>
            <p><b>Período:</b> {date_from.toString("dd/MM/yyyy")} até {date_to.toString("dd/MM/yyyy")}</p>

            <div class="section">
                <h2>Métricas Gerais</h2>
                <p>📦 Sacolas Inspecionadas: {total_bags}</p>
                <p>⚠️ Defeitos Identificados: {total_defects}</p>
                <p>📊 Taxa de Defeitos: {defect_rate:.2f}%</p>
                <p>🏷️ Defeito Mais Frequente: {top_defect}</p>
                <p>📅 Datas com Mais Defeitos: {dates_with_most_defects}</p>
            </div>

            <div class="section">
                <h2>Resumo por Tipo de Defeito</h2>
                <ul>{defects_by_type_html}</ul>
            </div>

            <div class="section">
                <h2>Resumo por Data</h2>
                <ul>{defects_by_date_html}</ul>
            </div>

            <div class="section">
                <h2>Registros Visuais</h2>
                <div style='display:flex; flex-wrap:wrap;'>{images_html}</div>
            </div>
        </body>
        </html>
        """

        # 🔹 Atualiza tela
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

        cadastros_dir = builtins.BASE_DIR / "cadastros"
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

    @staticmethod
    def generate_defect_summary(
        company: str, defect_summary_data: dict, date_from, date_to
    ):
        """
        Consolida/atualiza relatório diário em JSON.
        Sobrescreve os defeitos do dia atual para evitar duplicações.
        """

        from datetime import datetime
        import json

        cadastros_dir = builtins.BASE_DIR / "cadastros"
        company_dir = cadastros_dir / company
        documents_dir = company_dir / "documents"

        # Nome da pasta do dia
        day_folder_name = date_to.toString("yyyy-MM-dd")
        day_folder = documents_dir / day_folder_name
        day_folder.mkdir(parents=True, exist_ok=True)

        json_file = day_folder / f"summary_{day_folder_name}.json"

        # 🔹 Carregar JSON existente (se houver)
        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except Exception:
                existing_data = {}
        else:
            existing_data = {}

        defects_consolidated = existing_data.get("defects", {})

        # 🔹 Substituir os dados do dia atual pelos novos
        for date_str, bags in defect_summary_data.items():
            defects_consolidated[date_str] = bags  # sobrescreve completamente

        # 🔹 Salvar JSON atualizado
        final_data = {
            "company": company,
            "date_from": date_from.toString("dd/MM/yyyy"),
            "date_to": date_to.toString("dd/MM/yyyy"),
            "generated_at": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "defects": defects_consolidated,
        }

        try:
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(final_data, f, indent=4, ensure_ascii=False)
            print(f"✅ JSON consolidado salvo em {json_file}")
        except Exception as e:
            print(f"❌ Erro ao salvar JSON: {e}")

        return final_data
