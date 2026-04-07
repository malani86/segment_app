from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


class ImagePanel(QLabel):
    def __init__(self, title: str):
        super().__init__(title)
        self._title = title
        self._base_pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("border: 1px solid #888; background: #111; color: #ddd;")
        self.setWordWrap(True)

    def set_scaled_pixmap(self, pixmap: QPixmap) -> None:
        self._base_pixmap = pixmap
        self.setText("")
        self._refresh()

    def clear_to_title(self, title: str) -> None:
        self._title = title
        self._base_pixmap = None
        self.setPixmap(QPixmap())
        self.setText(title)

    def resizeEvent(self, event) -> None:
        self._refresh()
        super().resizeEvent(event)

    def _refresh(self) -> None:
        if self._base_pixmap is not None:
            scaled = self._base_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled)
            self.setText("")
        else:
            self.setPixmap(QPixmap())
            self.setText(self._title)
