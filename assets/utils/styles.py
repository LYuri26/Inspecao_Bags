MAIN_STYLE = """
/* ----- Base Geral ----- */
QWidget {
    font-size: 1em;  /* Unidade relativa para escalabilidade */
}

QMainWindow {
    background-color: #F5F5F5;
}

/* ----- Botões ----- */
QPushButton {
    background-color: #ee8d0a;
    color: #0e090e;
    border: none;
    padding: 0.5em 1em;  /* Unidades relativas */
    border-radius: 0.25em;
    font-weight: 600;
    min-height: 2em;
    font-size: 1.1em;  /* Tamanho relativo */
}
QPushButton:hover, QPushButton:focus {
    background-color: #632007;
    color: #fff;
    outline: none;
}

/* ----- Listas ----- */
QListWidget, QListView {
    background-color: white;
    border: 1px solid #ddd;
    border-radius: 0.25em;
    color: #7c6c60;
    font-size: 0.9em;  /* Tamanho relativo */
}

/* ----- Abas ----- */
QTabWidget::pane {
    border: 1px solid #ddd;
    border-radius: 0.25em;
    padding: 0.5em;
    background-color: white;
}
QTabBar::tab {
    background: #e0e0e0;
    padding: 0.5em 1.25em;
    border-top-left-radius: 0.25em;
    border-top-right-radius: 0.25em;
    color: #746464;
    font-weight: 600;
    min-width: 5em;
    margin-right: 0.125em;
    font-size: 0.95em;
}
QTabBar::tab:selected {
    background: white;
    border-bottom: 2px solid #ee8d0a;
    color: #632007;
    font-weight: 700;
}

/* ----- Campos Editáveis ----- */
QLineEdit, QComboBox, QSpinBox, QTextEdit, QTextBrowser {
    padding: 0.375em 0.5em;
    border: 1px solid #ddd;
    border-radius: 0.25em;
    color: #632007;
    background-color: white;
    font-size: 0.9em;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QTextEdit:focus {
    border-color: #ee8d0a;
    background-color: #fff9e6;
    outline: none;
}

/* ----- Labels ----- */
QLabel {
    color: #632007;
    font-weight: 600;
    font-size: 1em;  /* Tamanho base */
}

/* ----- Scrollbars ----- */
QScrollBar:vertical {
    background: #f0f0f0;
    width: 0.625em;
    margin: 0;
    border-radius: 0.25em;
}
QScrollBar::handle:vertical {
    background: #ee8d0a;
    min-height: 1.25em;
    border-radius: 0.25em;
}
QScrollBar::handle:vertical:hover {
    background: #632007;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

/* ----- Tooltips ----- */
QToolTip {
    background-color: #ee8d0a;
    color: #0e090e;
    border: none;
    padding: 0.3125em;
    border-radius: 0.25em;
    font-weight: bold;
    font-size: 0.9em;
}

/* ----- Componentes Específicos ----- */
#sidebar {
    font-size: 1.1em;  /* Tamanho aumentado para a sidebar */
}

#companyCombo {
    font-size: 0.95em;
    padding: 0.25em;
}

#statusLabel {
    font-weight: bold;
    font-size: 0.95em;
    padding: 0.125em;
    margin-top: 0;
    color: green;
}

#cameraLabel {
    background-color: transparent;
}

#alertPanel {
    border: none;
    background-color: transparent;
}

#resultsLabel {
    font-size: 1.05em;
    font-weight: bold;
    padding: 0.5em;
    margin-bottom: 0;
    background-color: transparent;
    color: #cc0000;
    border: none;
}
"""

ALERT_STYLES = {
    "info": "background-color: #D1ECF1; border-left: 4px solid #17A2B8; color: #0e090e;",
    "warning": "background-color: #FFF3CD; border-left: 4px solid #FFC107; color: #632007;",
    "error": "background-color: #F8D7DA; border-left: 4px solid #DC3545; color: #632007;",
    "success": "background-color: #D4EDDA; border-left: 4px solid #28A745; color: #0e090e;",
}
