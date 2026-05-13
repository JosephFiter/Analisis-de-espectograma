from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt


class CorrelationPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        title = QLabel("Pearson r  (Movement vs Audio)")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 11px; color: #aaa;")
        layout.addWidget(title)

        self._r_label = QLabel("r = —")
        self._r_label.setAlignment(Qt.AlignCenter)
        self._r_label.setStyleSheet(
            "font-size: 26px; font-weight: bold; color: #44aaff;"
        )
        layout.addWidget(self._r_label)

        self._interp_label = QLabel("Load both files to compute correlation")
        self._interp_label.setAlignment(Qt.AlignCenter)
        self._interp_label.setStyleSheet("font-size: 11px; color: #888;")
        layout.addWidget(self._interp_label)

    def set_correlation(self, r):
        if r is None or (isinstance(r, float) and r != r):  # NaN check
            self._r_label.setText("r = —")
            self._interp_label.setText("Load both files to compute correlation")
            return

        self._r_label.setText(f"r = {r:+.3f}")

        abs_r = abs(r)
        if abs_r < 0.1:
            strength = "No correlation"
        elif abs_r < 0.3:
            strength = "Weak"
        elif abs_r < 0.5:
            strength = "Moderate"
        elif abs_r < 0.7:
            strength = "Strong"
        else:
            strength = "Very strong"

        direction = "positive" if r >= 0 else "negative"
        self._interp_label.setText(f"{strength} {direction} correlation")
