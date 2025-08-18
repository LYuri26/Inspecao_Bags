import os
import json
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QCheckBox,
    QPushButton,
    QMessageBox,
    QHBoxLayout,
)
from PyQt5.QtCore import pyqtSignal
import re


class CompanyForm(QWidget):
    saved = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        self.form_layout = QFormLayout()
        self.layout.addLayout(self.form_layout)

        # Dados básicos
        self.name_input = QLineEdit()
        self.cnpj_input = QLineEdit()
        self.cnpj_input.setInputMask("99.999.999/9999-99")
        self.industry_input = QLineEdit()

        # Contato
        self.phone_input = QLineEdit()
        self.phone_input.setInputMask("(99) 9999-9999")
        self.mobile_input = QLineEdit()
        self.mobile_input.setInputMask("(99) 9 9999-9999")
        self.email_input = QLineEdit()

        # Endereço
        self.street_input = QLineEdit()
        self.number_input = QLineEdit()
        self.district_input = QLineEdit()
        self.state_input = QLineEdit()
        self.state_input.setMaxLength(2)
        self.state_input.setPlaceholderText("UF (2 letras)")
        self.country_input = QLineEdit()
        self.country_input.setText("Brasil")
        self.cep_input = QLineEdit()
        self.cep_input.setInputMask("99999-999")

        # Aceites
        self.accept_stains = QCheckBox("Aceitar manchas")
        self.accept_dirt = QCheckBox("Aceitar sujeiras")

        # Adicionando campos ao formulário
        self.form_layout.addRow("Nome*:", self.name_input)
        self.form_layout.addRow("CNPJ*:", self.cnpj_input)
        self.form_layout.addRow("Tipo de Indústria*:", self.industry_input)
        self.form_layout.addRow("Telefone:", self.phone_input)
        self.form_layout.addRow("Celular:", self.mobile_input)
        self.form_layout.addRow("E-mail:", self.email_input)
        self.form_layout.addRow("CEP:", self.cep_input)

        self.form_layout.addRow("Rua*:", self.street_input)

        address_number_layout = QHBoxLayout()
        address_number_layout.addWidget(self.number_input)
        address_number_layout.addWidget(self.district_input)
        self.form_layout.addRow("Número* / Bairro*:", address_number_layout)

        address_state_layout = QHBoxLayout()
        address_state_layout.addWidget(self.state_input)
        address_state_layout.addWidget(self.country_input)
        self.form_layout.addRow("Estado* (UF) / País*:", address_state_layout)

        self.form_layout.addRow("Aceites:", QWidget())  # Spacer
        self.form_layout.addRow(self.accept_stains)
        self.form_layout.addRow(self.accept_dirt)

    def validate_cnpj(self, cnpj):
        """Valida o CNPJ usando o algoritmo de dígitos verificadores."""
        # Remove caracteres não numéricos
        cnpj = "".join(filter(str.isdigit, cnpj))

        if len(cnpj) != 14:
            return False

        # Verifica se todos os dígitos são iguais
        if cnpj == cnpj[0] * 14:
            return False

        # Cálculo do primeiro dígito verificador
        sum = 0
        weight = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        for i in range(12):
            sum += int(cnpj[i]) * weight[i]

        rest = sum % 11
        digit1 = 0 if rest < 2 else 11 - rest

        # Cálculo do segundo dígito verificador
        sum = 0
        weight = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        for i in range(13):
            sum += int(cnpj[i]) * weight[i]

        rest = sum % 11
        digit2 = 0 if rest < 2 else 11 - rest

        # Verifica se os dígitos calculados conferem com os informados
        return int(cnpj[12]) == digit1 and int(cnpj[13]) == digit2

    def check_duplicate_cnpj(self, cnpj):
        """Verifica se já existe uma empresa com o mesmo CNPJ cadastrado."""
        base_dir = Path(__file__).resolve()
        while base_dir.name != "Inspecao_Bags":
            base_dir = base_dir.parent

        cadastros_dir = base_dir / "cadastros"

        if not cadastros_dir.exists():
            return False

        for empresa_dir in cadastros_dir.iterdir():
            if empresa_dir.is_dir():
                json_files = list(empresa_dir.glob("*.json"))
                if json_files:
                    with open(json_files[0], "r", encoding="utf-8") as f:
                        data = json.load(f)
                        existing_cnpj = "".join(
                            filter(str.isdigit, data.get("cnpj", ""))
                        )
                        current_cnpj = "".join(filter(str.isdigit, cnpj))
                        if existing_cnpj == current_cnpj:
                            return True
        return False

    def save_data(self):
        name = self.name_input.text().strip()
        cnpj = self.cnpj_input.text().strip()
        industry = self.industry_input.text().strip()
        street = self.street_input.text().strip()
        number = self.number_input.text().strip()
        district = self.district_input.text().strip()
        state = self.state_input.text().strip().upper()
        country = self.country_input.text().strip()
        cep = self.cep_input.text().strip()

        phone = self.phone_input.text().strip()
        mobile = self.mobile_input.text().strip()
        email = self.email_input.text().strip()

        # Validações obrigatórias
        if not name:
            QMessageBox.warning(self, "Erro", "Nome da empresa é obrigatório.")
            return

        if not self.validate_cnpj(cnpj):
            QMessageBox.warning(
                self, "Erro", "CNPJ inválido. Verifique o número digitado."
            )
            return

        if self.check_duplicate_cnpj(cnpj):
            QMessageBox.warning(
                self, "Erro", "Já existe uma empresa cadastrada com este CNPJ."
            )
            return

        if not all([industry, street, number, district, state, country]):
            QMessageBox.warning(
                self, "Erro", "Todos os campos de endereço são obrigatórios."
            )
            return

        if len(state) != 2:
            QMessageBox.warning(self, "Erro", "UF deve conter exatamente 2 letras.")
            return

        # Validação de e-mail
        if email and not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            QMessageBox.warning(self, "Erro", "E-mail inválido.")
            return

        # Validação de telefone fixo
        if phone and len("".join(filter(str.isdigit, phone))) != 10:
            QMessageBox.warning(self, "Erro", "Telefone fixo deve ter 10 dígitos.")
            return

        # Validação de celular
        if mobile and len("".join(filter(str.isdigit, mobile))) != 11:
            QMessageBox.warning(self, "Erro", "Celular deve ter 11 dígitos.")
            return

        # Montar o dicionário da empresa
        data = {
            "name": name,
            "cnpj": cnpj,
            "industry": industry,
            "address": {
                "street": street,
                "number": number,
                "district": district,
                "state": state,
                "country": country,
                "cep": cep,
            },
            "contact": {
                "phone": phone,
                "mobile": mobile,
                "email": email,
            },
            "policy": {
                "aceita_manchas": self.accept_stains.isChecked(),
                "aceita_sujeiras": self.accept_dirt.isChecked(),
                "aceita_rasgos": False,
                "aceita_cortes": False,
                "aceita_descosturas": False,
            },
        }

        # Caminho do diretório de cadastro
        base_dir = Path(__file__).resolve()
        while base_dir.name != "Inspecao_Bags":
            base_dir = base_dir.parent

        empresa_dir = base_dir / "cadastros" / name
        empresa_dir.mkdir(parents=True, exist_ok=True)

        json_path = empresa_dir / f"{name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        QMessageBox.information(self, "Sucesso", "Empresa cadastrada com sucesso.")
        self.saved.emit(data)

    def load_data(self, company_data):
        # Dados principais
        self.name_input.setText(company_data.get("name", ""))
        self.cnpj_input.setText(company_data.get("cnpj", ""))
        self.industry_input.setText(company_data.get("industry", ""))

        # Endereço
        address = company_data.get("address", {})
        self.street_input.setText(address.get("street", ""))
        self.number_input.setText(address.get("number", ""))
        self.district_input.setText(address.get("district", ""))
        self.state_input.setText(address.get("state", ""))
        self.country_input.setText(address.get("country", ""))
        self.cep_input.setText(address.get("cep", ""))

        # Contato
        contact = company_data.get("contact", {})
        self.phone_input.setText(contact.get("phone", ""))
        self.mobile_input.setText(contact.get("mobile", ""))
        self.email_input.setText(contact.get("email", ""))

        # Política de aceitação
        policy = company_data.get("policy", {})
        self.accept_stains.setChecked(policy.get("aceita_manchas", False))
        self.accept_dirt.setChecked(policy.get("aceita_sujeiras", False))
        # Campos extras de política, se desejar no futuro:
        # self.accept_tears.setChecked(policy.get("aceita_rasgos", False))
        # self.accept_cuts.setChecked(policy.get("aceita_cortes", False))
        # self.accept_unstitching.setChecked(policy.get("aceita_descosturas", False))

    def get_data(self):
        """Obtém e valida os dados do formulário"""
        # Obter valores dos campos
        name = self.name_input.text().strip()
        cnpj = self.cnpj_input.text().strip()
        industry = self.industry_input.text().strip()
        phone = self.phone_input.text().strip()
        mobile = self.mobile_input.text().strip()
        email = self.email_input.text().strip()
        cep = self.cep_input.text().strip()

        street = self.street_input.text().strip()
        number = self.number_input.text().strip()
        district = self.district_input.text().strip()
        state = self.state_input.text().strip().upper()
        country = self.country_input.text().strip()

        # Validações
        if not name:
            QMessageBox.warning(self, "Erro", "Nome da empresa é obrigatório.")
            return None

        if not self.validate_cnpj(cnpj):
            QMessageBox.warning(
                self, "Erro", "CNPJ inválido. Por favor, verifique o número digitado."
            )
            return None

        if not all([industry, street, number, district, state, country]):
            QMessageBox.warning(
                self,
                "Erro",
                "Todos os campos de endereço são obrigatórios.",
            )
            return None

        if len(state) != 2:
            QMessageBox.warning(
                self, "Erro", "O estado deve ser informado com 2 letras (UF)."
            )
            return None

        # Retornar estrutura completa de dados
        return {
            "name": name,
            "cnpj": cnpj,
            "industry": industry,
            "contact": {"phone": phone, "mobile": mobile, "email": email, "cep": cep},
            "address": {
                "street": street,
                "number": number,
                "district": district,
                "state": state,
                "country": country,
            },
            "policy": {
                "aceita_manchas": self.accept_stains.isChecked(),
                "aceita_sujeiras": self.accept_dirt.isChecked(),
                "aceita_rasgos": False,
                "aceita_cortes": False,
                "aceita_descosturas": False,
            },
        }
