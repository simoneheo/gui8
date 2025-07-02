# main.py
import sys
from PySide6.QtWidgets import QApplication
from main_window import MainWindowUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindowUI()
    window.show()
    sys.exit(app.exec())
