from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTableView,
    QHeaderView,
    QHBoxLayout,
    QPushButton,
    QDateEdit,
    QLabel,
)
from PyQt5.QtCore import QAbstractTableModel, Qt, QDate
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtGui import QTextDocument
from PyQt5.QtWidgets import QFileDialog


class HistoryTableModel(QAbstractTableModel):
    def __init__(self, data=None):
        super().__init__()
        self._data = data or []
        self._headers = ["Data", "Defeito", "Gravidade", "Empresa", "Resultado"]

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self._headers)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self._data[index.row()][index.column()]
        elif role == Qt.TextAlignmentRole:
            return Qt.AlignCenter
        elif role == Qt.ForegroundRole:
            if self._data[index.row()][4] == "REPROVADO":
                return Qt.red
            return Qt.green

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._headers[section]
        return super().headerData(section, orientation, role)

    def add_inspection(self, results):
        """Adiciona resultados de inspeção ao histórico"""
        for result in results:
            status = "APROVADO" if result["aprovado"] else "REPROVADO"
            self._data.append(
                [
                    QDate.currentDate().toString(Qt.ISODate),
                    result["tipo"].capitalize(),
                    str(result["gravidade"]),
                    "Empresa Atual",  # Substituir pelo nome real da empresa
                    status,
                ]
            )

        # Notifica a view que os dados mudaram
        self.layoutChanged.emit()

    def get_filtered_data(self, date_from, date_to):
        """Retorna os dados filtrados pelo período"""
        filtered_data = []
        for row in self._data:
            row_date = QDate.fromString(row[0], Qt.ISODate)
            if date_from <= row_date <= date_to:
                filtered_data.append(row)
        return filtered_data


class HistoryView(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        """Configura a interface do usuário"""
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        # Filtros
        filter_layout = QHBoxLayout()

        self.date_from = QDateEdit()
        self.date_from.setDate(QDate.currentDate().addDays(-7))
        self.date_to = QDateEdit()
        self.date_to.setDate(QDate.currentDate())
        self.filter_button = QPushButton("Filtrar")

        filter_layout.addWidget(QLabel("De:"))
        filter_layout.addWidget(self.date_from)
        filter_layout.addWidget(QLabel("Até:"))
        filter_layout.addWidget(self.date_to)
        filter_layout.addWidget(self.filter_button)

        self.layout.addLayout(filter_layout)

        # Tabela
        self.table = QTableView()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)

        self.model = HistoryTableModel()
        self.table.setModel(self.model)

        self.layout.addWidget(self.table)

        # Botão de exportação PDF
        export_layout = QHBoxLayout()
        self.export_pdf = QPushButton("Exportar PDF")
        export_layout.addWidget(self.export_pdf)
        self.layout.addLayout(export_layout)

        # Conexões
        self.filter_button.clicked.connect(self.filter_history)
        self.export_pdf.clicked.connect(self.export_to_pdf)

    def add_inspection(self, results):
        """Adiciona resultados de inspeção ao histórico"""
        self.model.add_inspection(results)

    def filter_history(self):
        """Filtra o histórico pelo período selecionado"""
        date_from = self.date_from.date()
        date_to = self.date_to.date()

        # Aqui você pode implementar a lógica de filtro real se necessário
        # Atualmente o modelo já filtra os dados quando solicitado para PDF

    def export_to_pdf(self):
        """Exporta o histórico filtrado para PDF"""
        date_from = self.date_from.date()
        date_to = self.date_to.date()
        filtered_data = self.model.get_filtered_data(date_from, date_to)

        if not filtered_data:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Exportar Histórico para PDF", "", "PDF Files (*.pdf)"
        )

        if file_name:
            # Cria o documento PDF
            doc = QTextDocument()
            html = self._generate_html_report(filtered_data, date_from, date_to)
            doc.setHtml(html)

            # Configura a impressora para PDF
            printer = QPrinter(QPrinter.HighResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(file_name)
            doc.print_(printer)

    def _generate_html_report(self, data, date_from, date_to):
        """Gera o conteúdo HTML para o relatório PDF"""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th {{ background-color: #4CAF50; color: white; padding: 8px; text-align: center; }}
                td {{ padding: 8px; border: 1px solid #ddd; text-align: center; }}
                .rejected {{ color: red; }}
                .approved {{ color: green; }}
            </style>
        </head>
        <body>
            <h1>Histórico de Inspeções</h1>
            <p>Período: {date_from.toString("dd/MM/yyyy")} a {date_to.toString("dd/MM/yyyy")}</p>
            <p>Total de registros: {len(data)}</p>
            
            <table>
                <tr>
                    <th>Data</th>
                    <th>Defeito</th>
                    <th>Gravidade</th>
                    <th>Empresa</th>
                    <th>Resultado</th>
                </tr>
        """

        for row in data:
            status_class = "rejected" if row[4] == "REPROVADO" else "approved"
            html += f"""
                <tr>
                    <td>{row[0]}</td>
                    <td>{row[1]}</td>
                    <td>{row[2]}</td>
                    <td>{row[3]}</td>
                    <td class='{status_class}'>{row[4]}</td>
                </tr>
            """

        html += """
            </table>
        </body>
        </html>
        """

        return html
