import sys
import os

# Ensure project root is on sys.path so sub-packages resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt

from ui.main_window import MainWindow


def _dark_palette(app: QApplication) -> QPalette:
    palette = QPalette()
    c = {
        QPalette.Window:          QColor(45,  45,  45),
        QPalette.WindowText:      QColor(220, 220, 220),
        QPalette.Base:            QColor(28,  28,  28),
        QPalette.AlternateBase:   QColor(38,  38,  38),
        QPalette.ToolTipBase:     QColor(55,  55,  55),
        QPalette.ToolTipText:     QColor(220, 220, 220),
        QPalette.Text:            QColor(220, 220, 220),
        QPalette.Button:          QColor(55,  55,  55),
        QPalette.ButtonText:      QColor(220, 220, 220),
        QPalette.BrightText:      QColor(255, 80,  80),
        QPalette.Highlight:       QColor(52,  120, 190),
        QPalette.HighlightedText: QColor(255, 255, 255),
        QPalette.Link:            QColor(80,  160, 230),
        QPalette.Disabled + QPalette.Text:       QColor(100, 100, 100),
        QPalette.Disabled + QPalette.ButtonText: QColor(100, 100, 100),
    }
    for role, color in c.items():
        palette.setColor(role, color)
    return palette


def main():
    # High-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Ayudantia Itba")
    app.setOrganizationName("ITBA")
    app.setStyle("Fusion")
    app.setPalette(_dark_palette(app))

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
