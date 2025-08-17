from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QButtonGroup
from PyQt5.QtCore import pyqtSignal


class Sidebar(QWidget):
    view_changed = pyqtSignal(str)
    open_monitor_window = pyqtSignal()  # Novo sinal para abrir janela de monitoramento

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        # Button group
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)

        # Buttons
        self.buttons = {
            "camera": QPushButton("Monitoramento"),
            "companies": QPushButton("Empresas"),
            "reports": QPushButton("Relatórios"),
            "history": QPushButton("Histórico"),
        }

        # Add buttons to layout and group
        for name, btn in self.buttons.items():
            btn.setCheckable(True)
            self.layout.addWidget(btn)
            self.button_group.addButton(btn)

            # Modificação aqui para tratar o botão de monitoramento diferente
            if name == "camera":
                btn.clicked.connect(self._handle_monitor_click)
            else:
                btn.clicked.connect(lambda _, n=name: self.view_changed.emit(n))

        # Set first button as checked
        self.buttons["camera"].setChecked(True)
        self.layout.addStretch()

    def _handle_monitor_click(self):
        """Manipulador especial para o botão de monitoramento"""
        self.open_monitor_window.emit()  # Emite o novo sinal
        self.buttons["camera"].setChecked(False)  # Mantém o botão desmarcado
