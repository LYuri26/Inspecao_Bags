from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette


class AlertPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        # Title
        self.title = QLabel("Alertas")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.layout.addWidget(self.title)

        # Scroll area for alerts
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.scroll_content)
        self.layout.addWidget(self.scroll)

    def add_alert(self, message, severity=1):
        """Add an alert with color coding based on severity"""
        alert = QLabel(message)
        alert.setWordWrap(True)
        alert.setMargin(5)
        
        # Set color based on severity
        if severity == 1:
            alert.setStyleSheet("background-color: #FFF3CD; border-left: 4px solid #FFC107;")
        elif severity == 2:
            alert.setStyleSheet("background-color: #F8D7DA; border-left: 4px solid #DC3545;")
        else:
            alert.setStyleSheet("background-color: #D4EDDA; border-left: 4px solid #28A745;")
            
        self.scroll_layout.addWidget(alert)