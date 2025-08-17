import os
import json
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QTabWidget,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QMessageBox,
    QPushButton,
    QLineEdit,
)
from PyQt5.QtCore import pyqtSignal, Qt
from src.ui.widgets.company_form import CompanyForm


class CompaniesView(QWidget):
    policy_updated = pyqtSignal(object)
    company_selected = pyqtSignal(object)
    company_deleted = pyqtSignal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.current_company = None
        self.is_editing = False

        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.base_path = os.path.join(BASE_DIR, "cadastros")
        os.makedirs(self.base_path, exist_ok=True)

        self.companies = []
        self.setup_ui()
        self.load_companies_from_disk()

    def setup_ui(self):
        self.layout = QHBoxLayout(self)
        self.setLayout(self.layout)

        list_layout = QVBoxLayout()

        # Adiciona barra de pesquisa
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Pesquisar empresa...")
        self.search_input.textChanged.connect(self.filter_companies)
        list_layout.addWidget(self.search_input)

        self.setup_buttons(list_layout)

        self.company_list = QListWidget()
        self.company_list.itemSelectionChanged.connect(self.on_company_selected)
        list_layout.addWidget(self.company_list)

        self.layout.addLayout(list_layout, stretch=1)

        self.details_area = QTabWidget()
        self.details_area.setTabEnabled(
            1, False
        )  # Aba de edição desabilitada por padrão
        self.setup_details_tabs()
        self.layout.addWidget(self.details_area, stretch=3)

    def filter_companies(self, text):
        """Filtra a lista de empresas conforme o texto digitado"""
        if not text:
            # Mostra todas as empresas se a busca estiver vazia
            for i in range(self.company_list.count()):
                self.company_list.item(i).setHidden(False)
            return

        text = text.lower()
        for i in range(self.company_list.count()):
            item = self.company_list.item(i)
            company_name = item.text().lower()
            item.setHidden(text not in company_name)

    def setup_buttons(self, layout):
        self.btn_new = QPushButton("Nova Empresa")
        self.btn_edit = QPushButton("Editar")
        self.btn_delete = QPushButton("Excluir")
        self.btn_save = QPushButton("Salvar")
        self.btn_cancel = QPushButton("Cancelar")

        self.btn_new.clicked.connect(self.new_company)
        self.btn_edit.clicked.connect(self.start_edit)
        self.btn_delete.clicked.connect(self.delete_company)
        self.btn_save.clicked.connect(self.save_company)
        self.btn_cancel.clicked.connect(self.cancel_edit)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_new)
        btn_layout.addWidget(self.btn_edit)
        btn_layout.addWidget(self.btn_delete)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

        self.update_buttons_state()

    def setup_details_tabs(self):
        self.view_tab = QWidget()
        self.setup_view_tab()
        self.details_area.addTab(self.view_tab, "Detalhes")

        self.edit_tab = QWidget()
        self.setup_edit_tab()
        self.details_area.addTab(self.edit_tab, "Editar")
        self.details_area.setTabEnabled(1, False)
        self.details_area.setCurrentIndex(0)

    def setup_view_tab(self):
        layout = QVBoxLayout(self.view_tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.view_layout = QVBoxLayout(content)

        self.company_name = QLabel()
        self.company_name.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.company_cnpj = QLabel()
        self.company_industry = QLabel()

        # Novos campos para contato
        self.company_phone = QLabel()
        self.company_mobile = QLabel()
        self.company_email = QLabel()
        self.company_cep = QLabel()

        self.company_address = QLabel()
        self.policy_info = QLabel()
        self.policy_info.setWordWrap(True)

        # Adicionando widgets ao layout
        self.view_layout.addWidget(self.company_name)
        self.view_layout.addWidget(QLabel("CNPJ:"))
        self.view_layout.addWidget(self.company_cnpj)
        self.view_layout.addWidget(QLabel("Indústria:"))
        self.view_layout.addWidget(self.company_industry)

        # Adicionando seção de contato
        self.view_layout.addWidget(QLabel("\nContato:"))
        self.view_layout.addWidget(QLabel("Telefone:"))
        self.view_layout.addWidget(self.company_phone)
        self.view_layout.addWidget(QLabel("Celular:"))
        self.view_layout.addWidget(self.company_mobile)
        self.view_layout.addWidget(QLabel("E-mail:"))
        self.view_layout.addWidget(self.company_email)
        self.view_layout.addWidget(QLabel("CEP:"))
        self.view_layout.addWidget(self.company_cep)

        # Endereço
        self.view_layout.addWidget(QLabel("\nEndereço:"))
        self.view_layout.addWidget(self.company_address)

        # Políticas
        self.view_layout.addWidget(QLabel("\nPolíticas:"))
        self.view_layout.addWidget(self.policy_info)

        self.view_layout.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def setup_edit_tab(self):
        edit_layout = QVBoxLayout()
        edit_layout.setContentsMargins(10, 10, 10, 10)
        self.company_form = CompanyForm()
        edit_layout.addWidget(self.company_form)
        edit_layout.addItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )
        self.edit_tab.setLayout(edit_layout)

    def load_companies_from_disk(self):
        self.companies.clear()
        self.company_list.clear()

        for company_folder in os.listdir(self.base_path):
            folder_path = os.path.join(self.base_path, company_folder)
            if os.path.isdir(folder_path):
                json_file = os.path.join(folder_path, f"{company_folder}.json")
                if os.path.isfile(json_file):
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            company = json.load(f)
                            self.companies.append(company)
                    except Exception as e:
                        print(f"Erro ao carregar {json_file}: {e}")

        self.populate_company_list()
        # Atualiza a câmera também
        if hasattr(self.parent, "views") and "camera" in self.parent.views:
            self.parent.views["camera"].set_companies(self.companies)

    def populate_company_list(self):
        self.company_list.clear()
        for company in sorted(self.companies, key=lambda x: x["name"]):
            item = QListWidgetItem(company["name"])
            item.setData(Qt.UserRole, company)
            self.company_list.addItem(item)

        # Update camera view if it exists in parent
        if hasattr(self.parent, "views") and "camera" in self.parent.views:
            self.parent.views["camera"].set_companies(self.companies)

    def on_company_selected(self):
        selected = self.company_list.currentItem()
        if not selected:
            return

        self.current_company = selected.data(Qt.UserRole)
        self.update_view_tab()
        self.company_selected.emit(self.current_company)
        self.details_area.setCurrentIndex(0)
        self.details_area.setTabEnabled(1, False)
        self.update_buttons_state()

    def update_view_tab(self):
        if not self.current_company:
            self.clear_view_tab()
            return

        # Dados básicos
        self.company_name.setText(self.current_company["name"])
        self.company_cnpj.setText(self.current_company.get("cnpj", "Não informado"))
        self.company_industry.setText(
            self.current_company.get("industry", "Não informado")
        )

        # Contato
        contact = self.current_company.get("contact", {})
        self.company_phone.setText(contact.get("phone", "Não informado"))
        self.company_mobile.setText(contact.get("mobile", "Não informado"))
        self.company_email.setText(contact.get("email", "Não informado"))
        self.company_cep.setText(contact.get("cep", "Não informado"))

        # Endereço
        address = self.current_company.get("address", {})
        address_text = (
            f"{address.get('street', '')}, {address.get('number', '')}\n"
            f"Bairro: {address.get('district', '')}\n"
            f"{address.get('state', '')}/{address.get('country', '')}"
        )
        self.company_address.setText(address_text)

        # Políticas
        policy = self.current_company.get("policy", {})
        policy_text = [
            f"- {key.replace('aceita_', '').capitalize()}: {'Sim' if val else 'Não'}"
            for key, val in policy.items()
        ]
        self.policy_info.setText("\n".join(policy_text))

    def clear_view_tab(self):
        self.company_name.setText("")
        self.company_cnpj.setText("")
        self.company_industry.setText("")
        self.company_phone.setText("")
        self.company_mobile.setText("")
        self.company_email.setText("")
        self.company_cep.setText("")
        self.company_address.setText("")
        self.policy_info.setText("")

    def new_company(self):
        self.is_editing = True
        self.current_company = None
        self.company_form.load_data({})
        self.details_area.setTabEnabled(1, True)
        self.details_area.setCurrentIndex(1)
        self.update_buttons_state()

    def start_edit(self):
        if not self.current_company:
            QMessageBox.warning(
                self, "Aviso", "Nenhuma empresa selecionada para edição"
            )
            return

        self.is_editing = True
        self.company_form.load_data(self.current_company)
        self.details_area.setTabEnabled(1, True)
        self.details_area.setCurrentIndex(1)
        self.update_buttons_state()

    def delete_company(self):
        if not self.current_company:
            QMessageBox.warning(
                self, "Aviso", "Nenhuma empresa selecionada para exclusão"
            )
            return

        reply = QMessageBox.question(
            self,
            "Confirmação",
            f"Tem certeza que deseja excluir a empresa {self.current_company['name']}?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.perform_company_deletion()

    def perform_company_deletion(self):
        company_name = self.current_company["name"]
        company_name_safe = "".join(
            c for c in company_name if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        company_folder = os.path.join(self.base_path, company_name_safe)

        try:
            for file in os.listdir(company_folder):
                os.remove(os.path.join(company_folder, file))
            os.rmdir(company_folder)

            self.companies = [c for c in self.companies if c["name"] != company_name]
            self.populate_company_list()
            self.current_company = None
            self.update_view_tab()

            QMessageBox.information(self, "Sucesso", "Empresa excluída com sucesso!")
            self.company_deleted.emit(company_name)

        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha ao excluir empresa: {str(e)}")

    def save_company(self):
        company_data = self.company_form.get_data()
        if not company_data:
            return

        if self.current_company is None:
            self.create_new_company(company_data)
        else:
            self.update_existing_company(company_data)

        self.save_to_disk(company_data)

    def create_new_company(self, company_data):
        self.current_company = company_data
        self.companies.append(company_data)
        item = QListWidgetItem(company_data["name"])
        item.setData(Qt.UserRole, company_data)
        self.company_list.addItem(item)
        self.company_list.setCurrentItem(item)

    def update_existing_company(self, company_data):
        self.current_company.update(company_data)
        self.company_list.currentItem().setText(company_data["name"])

    def save_to_disk(self, company_data):
        company_name_safe = "".join(
            c for c in company_data["name"] if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        company_folder = os.path.join(self.base_path, company_name_safe)
        os.makedirs(company_folder, exist_ok=True)

        file_path = os.path.join(company_folder, f"{company_name_safe}.json")

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.current_company, f, ensure_ascii=False, indent=4)

            self.is_editing = False
            self.details_area.setCurrentIndex(0)
            self.details_area.setTabEnabled(1, False)
            self.update_view_tab()
            self.update_buttons_state()

            QMessageBox.information(
                self, "Sucesso", "Dados da empresa salvos com sucesso!"
            )
            self.policy_updated.emit(self.current_company.get("policy", {}))

        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha ao salvar arquivo:\n{str(e)}")

    def cancel_edit(self):
        self.is_editing = False
        self.details_area.setCurrentIndex(0)
        self.details_area.setTabEnabled(1, False)
        self.update_buttons_state()

        current_item = self.company_list.currentItem()
        if current_item:
            self.current_company = current_item.data(Qt.UserRole)
            self.update_view_tab()
        else:
            self.current_company = None

    def update_buttons_state(self):
        has_selection = self.current_company is not None
        is_editing = self.is_editing

        self.btn_new.setVisible(not is_editing)
        self.btn_edit.setVisible(not is_editing and has_selection)
        self.btn_delete.setVisible(not is_editing and has_selection)
        self.btn_save.setVisible(is_editing)
        self.btn_cancel.setVisible(is_editing)
