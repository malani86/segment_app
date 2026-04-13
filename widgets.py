from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QStyle, QStyleOption


class ImagePanel(QLabel):
    def __init__(self, title: str):
        super().__init__(title)
        self._title = title
        self._base_pixmap: QPixmap | None = None
        self._display_pixmap = QPixmap()
        self._zoom_factor = 1.0
        self._pan_offset = QPoint(0, 0)
        self._last_drag_pos: QPoint | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(160, 160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("border: 1px solid #888; background: #111; color: #ddd;")
        self.setWordWrap(True)
        self.setToolTip("Use the mouse wheel to zoom. Drag to move in the image. Double-click to reset.")

    def set_scaled_pixmap(self, pixmap: QPixmap) -> None:
        self._base_pixmap = pixmap
        self._display_pixmap = QPixmap()
        self._zoom_factor = 1.0
        self._pan_offset = QPoint(0, 0)
        self.setText("")
        self._refresh()

    def clear_to_title(self, title: str) -> None:
        self._title = title
        self._base_pixmap = None
        self._display_pixmap = QPixmap()
        self._zoom_factor = 1.0
        self._pan_offset = QPoint(0, 0)
        self._last_drag_pos = None
        self.setPixmap(QPixmap())
        self.setText(title)
        self.unsetCursor()

    def resizeEvent(self, event) -> None:
        self._refresh()
        super().resizeEvent(event)

    def wheelEvent(self, event) -> None:
        if self._base_pixmap is None:
            super().wheelEvent(event)
            return

        angle_delta = event.angleDelta().y()
        if angle_delta == 0:
            return

        zoom_step = 1.15 if angle_delta > 0 else 1 / 1.15
        self._zoom_factor = max(0.25, min(6.0, self._zoom_factor * zoom_step))
        self._refresh()
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and not self._display_pixmap.isNull():
            self._last_drag_pos = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._last_drag_pos is not None and not self._display_pixmap.isNull():
            current_pos = event.position().toPoint()
            delta = current_pos - self._last_drag_pos
            self._last_drag_pos = current_pos
            self._pan_offset += delta
            self._clamp_pan_offset()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._last_drag_pos is not None:
            self._last_drag_pos = None
            self._update_cursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if self._base_pixmap is not None:
            self._zoom_factor = 1.0
            self._pan_offset = QPoint(0, 0)
            self._refresh()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event) -> None:
        if self._display_pixmap.isNull():
            super().paintEvent(event)
            return

        option = QStyleOption()
        option.initFrom(self)
        painter = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, option, painter, self)

        x = (self.width() - self._display_pixmap.width()) // 2 + self._pan_offset.x()
        y = (self.height() - self._display_pixmap.height()) // 2 + self._pan_offset.y()
        painter.drawPixmap(x, y, self._display_pixmap)

    def _refresh(self) -> None:
        if self._base_pixmap is not None:
            target_size = self.size() * self._zoom_factor
            self._display_pixmap = self._base_pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._clamp_pan_offset()
            self.setText("")
            self.update()
        else:
            self._display_pixmap = QPixmap()
            self.setText(self._title)
            self.update()

        self._update_cursor()

    def _clamp_pan_offset(self) -> None:
        if self._display_pixmap.isNull():
            self._pan_offset = QPoint(0, 0)
            return

        max_x = max(0, (self._display_pixmap.width() - self.width()) // 2)
        max_y = max(0, (self._display_pixmap.height() - self.height()) // 2)
        self._pan_offset.setX(max(-max_x, min(max_x, self._pan_offset.x())))
        self._pan_offset.setY(max(-max_y, min(max_y, self._pan_offset.y())))

    def _update_cursor(self) -> None:
        if self._display_pixmap.isNull():
            self.unsetCursor()
            return

        can_pan = self._display_pixmap.width() > self.width() or self._display_pixmap.height() > self.height()
        if self._last_drag_pos is not None:
            self.setCursor(Qt.ClosedHandCursor)
        elif can_pan:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.unsetCursor()
