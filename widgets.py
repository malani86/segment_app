from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QDialogButtonBox,
    QDoubleSpinBox,
    QLineEdit,
    QSlider,
    QSpinBox,
    QStyle,
    QStyleOption,
    QVBoxLayout,
    QWidget,
)

from app_models import (
    STACK_VIEW_PROJECTION,
    STACK_VIEW_SLICE,
    TIFF_MODE_ALL_SLICES,
    TIFF_MODE_CURRENT_SLICE,
    TIFF_MODE_MAX_PROJECTION,
    VIEWER_MODE_MASK,
    VIEWER_MODE_ORIGINAL,
    VIEWER_MODE_OVERLAY,
)
from app_models import AppSettings


class ImagePanel(QLabel):
    zoomChanged = Signal(float)
    fitModeChanged = Signal(bool)

    def __init__(self, title: str):
        super().__init__(title)
        self._title = title
        self._base_pixmap: QPixmap | None = None
        self._display_pixmap = QPixmap()
        self._zoom_factor = 1.0
        self._pan_offset = QPoint(0, 0)
        self._last_drag_pos: QPoint | None = None
        self._fit_to_window = True
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(240, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("border: 1px solid #888; background: #111; color: #ddd;")
        self.setWordWrap(True)
        self.setToolTip("Use the mouse wheel to zoom. Drag to move in the image. Double-click to reset.")

    def set_scaled_pixmap(self, pixmap: QPixmap) -> None:
        self._base_pixmap = pixmap
        self._display_pixmap = QPixmap()
        self._pan_offset = QPoint(0, 0)
        self.setText("")
        if self._fit_to_window:
            self._refresh()
        else:
            self._apply_zoom(self._zoom_factor)

    def clear_to_title(self, title: str) -> None:
        self._title = title
        self._base_pixmap = None
        self._display_pixmap = QPixmap()
        self._zoom_factor = 1.0
        self._pan_offset = QPoint(0, 0)
        self._last_drag_pos = None
        self._fit_to_window = True
        self.setPixmap(QPixmap())
        self.setText(title)
        self.unsetCursor()
        self.zoomChanged.emit(self._zoom_factor)
        self.fitModeChanged.emit(self._fit_to_window)

    def has_image(self) -> bool:
        return self._base_pixmap is not None and not self._base_pixmap.isNull()

    def is_fit_to_window(self) -> bool:
        return self._fit_to_window

    def fit_to_window(self) -> None:
        self._fit_to_window = True
        self._pan_offset = QPoint(0, 0)
        self._refresh()
        self.fitModeChanged.emit(True)

    def reset_zoom(self) -> None:
        self._fit_to_window = False
        self._pan_offset = QPoint(0, 0)
        self._apply_zoom(1.0)
        self.fitModeChanged.emit(False)

    def zoom_in(self) -> None:
        self._step_zoom(1.15)

    def zoom_out(self) -> None:
        self._step_zoom(1 / 1.15)

    def resizeEvent(self, event) -> None:
        self._refresh()
        super().resizeEvent(event)

    def wheelEvent(self, event) -> None:
        if not self.has_image():
            super().wheelEvent(event)
            return

        angle_delta = event.angleDelta().y()
        if angle_delta == 0:
            return

        self._step_zoom(1.15 if angle_delta > 0 else 1 / 1.15)
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
        if self.has_image():
            self.fit_to_window()
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

    def _step_zoom(self, multiplier: float) -> None:
        self._fit_to_window = False
        self.fitModeChanged.emit(False)
        self._apply_zoom(self._zoom_factor * multiplier)

    def _apply_zoom(self, zoom_factor: float) -> None:
        self._zoom_factor = max(0.1, min(12.0, zoom_factor))
        self._refresh()
        self.zoomChanged.emit(self._zoom_factor)

    def _refresh(self) -> None:
        if self._base_pixmap is not None:
            if self._fit_to_window:
                target_size = self.size()
            else:
                target_size = self._base_pixmap.size() * self._zoom_factor
            self._display_pixmap = self._base_pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._clamp_pan_offset()
            self.setText("")
            self.update()
            if self._fit_to_window and not self._base_pixmap.isNull():
                fit_zoom = min(
                    self.width() / max(1, self._base_pixmap.width()),
                    self.height() / max(1, self._base_pixmap.height()),
                )
                self._zoom_factor = max(0.01, fit_zoom)
                self.zoomChanged.emit(self._zoom_factor)
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


class InspectionViewer(QWidget):
    modeChanged = Signal(str)
    fitModeChanged = Signal(bool)
    sliceIndexChanged = Signal(int)
    stackViewModeChanged = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._images: dict[str, QPixmap | None] = {
            VIEWER_MODE_ORIGINAL: None,
            VIEWER_MODE_MASK: None,
            VIEWER_MODE_OVERLAY: None,
        }
        self._titles: dict[str, str] = {
            VIEWER_MODE_ORIGINAL: "Original image",
            VIEWER_MODE_MASK: "Predicted mask",
            VIEWER_MODE_OVERLAY: "Overlay",
        }
        self._source_mode = ""
        self._slice_summary = "Single image"
        self._slice_count = 1
        self._slice_index = 0
        self._is_stack = False
        self._stack_view_mode = STACK_VIEW_SLICE
        self._build_ui()
        self._connect_signals()
        self.set_mode(VIEWER_MODE_ORIGINAL)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        toolbar = QFrame()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(6)

        self.original_btn = QPushButton("Original")
        self.mask_btn = QPushButton("Mask")
        self.overlay_btn = QPushButton("Overlay")
        for button in (self.original_btn, self.mask_btn, self.overlay_btn):
            button.setCheckable(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.original_btn)
        self.mode_group.addButton(self.mask_btn)
        self.mode_group.addButton(self.overlay_btn)

        self.fit_btn = QPushButton("Fit")
        self.actual_size_btn = QPushButton("100%")
        self.zoom_in_btn = QPushButton("+")
        self.zoom_out_btn = QPushButton("-")

        self.zoom_label = QLabel("100%")
        self.info_label = QLabel("No image selected")
        self.info_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        toolbar_layout.addWidget(self.original_btn)
        toolbar_layout.addWidget(self.mask_btn)
        toolbar_layout.addWidget(self.overlay_btn)
        toolbar_layout.addSpacing(12)
        toolbar_layout.addWidget(self.fit_btn)
        toolbar_layout.addWidget(self.actual_size_btn)
        toolbar_layout.addWidget(self.zoom_out_btn)
        toolbar_layout.addWidget(self.zoom_in_btn)
        toolbar_layout.addWidget(self.zoom_label)
        toolbar_layout.addStretch(1)
        toolbar_layout.addWidget(self.info_label)

        self.canvas = ImagePanel("Select an image to inspect.")
        self.canvas.setMinimumSize(420, 320)

        stack_bar = QFrame()
        stack_layout = QHBoxLayout(stack_bar)
        stack_layout.setContentsMargins(0, 0, 0, 0)
        stack_layout.setSpacing(6)

        self.slice_view_btn = QPushButton("Slice")
        self.projection_view_btn = QPushButton("Max projection")
        self.slice_view_btn.setCheckable(True)
        self.projection_view_btn.setCheckable(True)
        self.stack_view_group = QButtonGroup(self)
        self.stack_view_group.setExclusive(True)
        self.stack_view_group.addButton(self.slice_view_btn)
        self.stack_view_group.addButton(self.projection_view_btn)

        self.prev_slice_btn = QPushButton("Prev slice")
        self.next_slice_btn = QPushButton("Next slice")
        self.slice_label = QLabel("Slice 1/1")
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setRange(0, 0)

        stack_layout.addWidget(self.slice_view_btn)
        stack_layout.addWidget(self.projection_view_btn)
        stack_layout.addSpacing(12)
        stack_layout.addWidget(self.prev_slice_btn)
        stack_layout.addWidget(self.slice_slider, 1)
        stack_layout.addWidget(self.next_slice_btn)
        stack_layout.addWidget(self.slice_label)

        root.addWidget(toolbar)
        root.addWidget(stack_bar)
        root.addWidget(self.canvas, 1)

    def _connect_signals(self) -> None:
        self.original_btn.clicked.connect(lambda: self.set_mode(VIEWER_MODE_ORIGINAL))
        self.mask_btn.clicked.connect(lambda: self.set_mode(VIEWER_MODE_MASK))
        self.overlay_btn.clicked.connect(lambda: self.set_mode(VIEWER_MODE_OVERLAY))
        self.fit_btn.clicked.connect(self.canvas.fit_to_window)
        self.actual_size_btn.clicked.connect(self.canvas.reset_zoom)
        self.zoom_in_btn.clicked.connect(self.canvas.zoom_in)
        self.zoom_out_btn.clicked.connect(self.canvas.zoom_out)
        self.canvas.zoomChanged.connect(self._update_zoom_label)
        self.canvas.fitModeChanged.connect(self.fitModeChanged.emit)
        self.prev_slice_btn.clicked.connect(self._go_to_previous_slice)
        self.next_slice_btn.clicked.connect(self._go_to_next_slice)
        self.slice_slider.valueChanged.connect(self._emit_slice_index_changed)
        self.slice_view_btn.clicked.connect(lambda: self._set_stack_view_mode(STACK_VIEW_SLICE))
        self.projection_view_btn.clicked.connect(lambda: self._set_stack_view_mode(STACK_VIEW_PROJECTION))

    def set_images(
        self,
        *,
        original: QPixmap | None,
        mask: QPixmap | None,
        overlay: QPixmap | None,
        source_mode: str = "",
        is_stack: bool = False,
        available_slices: int = 1,
        current_slice_index: int = 0,
        stack_view_mode: str = STACK_VIEW_SLICE,
    ) -> None:
        self._images = {
            VIEWER_MODE_ORIGINAL: original if original and not original.isNull() else None,
            VIEWER_MODE_MASK: mask if mask and not mask.isNull() else None,
            VIEWER_MODE_OVERLAY: overlay if overlay and not overlay.isNull() else None,
        }
        self._source_mode = source_mode
        self._is_stack = is_stack
        self._slice_count = max(1, available_slices)
        self._slice_index = max(0, min(self._slice_count - 1, current_slice_index))
        self._stack_view_mode = stack_view_mode
        self._sync_stack_controls()
        self._slice_summary = self._format_slice_summary(available_slices, current_slice_index)
        self._update_mode_buttons()
        current_mode = self.current_mode()
        if self._images[current_mode] is None:
            current_mode = self._first_available_mode()
        self.set_mode(current_mode)

    def clear(self, title: str = "Select an image to inspect.") -> None:
        self._images = {
            VIEWER_MODE_ORIGINAL: None,
            VIEWER_MODE_MASK: None,
            VIEWER_MODE_OVERLAY: None,
        }
        self._source_mode = ""
        self._slice_summary = "Single image"
        self._is_stack = False
        self._slice_count = 1
        self._slice_index = 0
        self._stack_view_mode = STACK_VIEW_SLICE
        self.canvas.clear_to_title(title)
        self._sync_stack_controls()
        self._update_mode_buttons()
        self._update_info_label()

    def set_mode(self, mode: str) -> None:
        pixmap = self._images.get(mode)
        if pixmap is None:
            self.canvas.clear_to_title(f"{self._titles[mode]} not available.")
        else:
            self.canvas.set_scaled_pixmap(pixmap)
        self.original_btn.setChecked(mode == VIEWER_MODE_ORIGINAL)
        self.mask_btn.setChecked(mode == VIEWER_MODE_MASK)
        self.overlay_btn.setChecked(mode == VIEWER_MODE_OVERLAY)
        self._update_mode_buttons()
        self._update_info_label(mode)
        self.modeChanged.emit(mode)

    def current_mode(self) -> str:
        if self.mask_btn.isChecked():
            return VIEWER_MODE_MASK
        if self.overlay_btn.isChecked():
            return VIEWER_MODE_OVERLAY
        return VIEWER_MODE_ORIGINAL

    def show_original(self) -> None:
        self.set_mode(VIEWER_MODE_ORIGINAL)

    def show_mask(self) -> None:
        self.set_mode(VIEWER_MODE_MASK)

    def show_overlay(self) -> None:
        self.set_mode(VIEWER_MODE_OVERLAY)

    def set_stack_state(
        self,
        *,
        is_stack: bool,
        available_slices: int,
        current_slice_index: int,
        stack_view_mode: str,
    ) -> None:
        self._is_stack = is_stack
        self._slice_count = max(1, available_slices)
        self._slice_index = max(0, min(self._slice_count - 1, current_slice_index))
        self._stack_view_mode = stack_view_mode
        self._slice_summary = self._format_slice_summary(self._slice_count, self._slice_index)
        self._sync_stack_controls()
        self._update_info_label()

    def _first_available_mode(self) -> str:
        for mode in (VIEWER_MODE_ORIGINAL, VIEWER_MODE_MASK, VIEWER_MODE_OVERLAY):
            if self._images.get(mode) is not None:
                return mode
        return VIEWER_MODE_ORIGINAL

    def _update_mode_buttons(self) -> None:
        self.original_btn.setEnabled(self._images[VIEWER_MODE_ORIGINAL] is not None)
        self.mask_btn.setEnabled(self._images[VIEWER_MODE_MASK] is not None)
        self.overlay_btn.setEnabled(self._images[VIEWER_MODE_OVERLAY] is not None)
        self.fit_btn.setEnabled(self.canvas.has_image())
        self.actual_size_btn.setEnabled(self.canvas.has_image())
        self.zoom_in_btn.setEnabled(self.canvas.has_image())
        self.zoom_out_btn.setEnabled(self.canvas.has_image())

    def _update_zoom_label(self, zoom_factor: float) -> None:
        self.zoom_label.setText(f"{int(round(zoom_factor * 100))}%")

    def _update_info_label(self, mode: str | None = None) -> None:
        selected_mode = mode or self.current_mode()
        title = self._titles[selected_mode]
        details = [title]
        if self._source_mode:
            details.append(self._source_mode)
        if self._slice_summary:
            details.append(self._slice_summary)
        self.info_label.setText(" | ".join(details))
        self._update_mode_buttons()

    def _format_slice_summary(self, available_slices: int, current_slice_index: int) -> str:
        if available_slices <= 1:
            return "Single image"
        if self._stack_view_mode == STACK_VIEW_PROJECTION:
            return f"Stack: max projection from {available_slices} slices"
        return f"Stack: slice {current_slice_index + 1}/{available_slices}"

    def _sync_stack_controls(self) -> None:
        is_stack = self._is_stack and self._slice_count > 1
        self.slice_view_btn.setEnabled(is_stack)
        self.projection_view_btn.setEnabled(is_stack)
        self.prev_slice_btn.setEnabled(is_stack and self._stack_view_mode == STACK_VIEW_SLICE and self._slice_index > 0)
        self.next_slice_btn.setEnabled(
            is_stack and self._stack_view_mode == STACK_VIEW_SLICE and self._slice_index < self._slice_count - 1
        )
        self.slice_slider.setEnabled(is_stack and self._stack_view_mode == STACK_VIEW_SLICE)
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, max(0, self._slice_count - 1))
        self.slice_slider.setValue(self._slice_index)
        self.slice_slider.blockSignals(False)
        self.slice_view_btn.setChecked(self._stack_view_mode == STACK_VIEW_SLICE)
        self.projection_view_btn.setChecked(self._stack_view_mode == STACK_VIEW_PROJECTION)
        if is_stack:
            if self._stack_view_mode == STACK_VIEW_PROJECTION:
                self.slice_label.setText(f"Projection ({self._slice_count} slices)")
            else:
                self.slice_label.setText(f"Slice {self._slice_index + 1}/{self._slice_count}")
        else:
            self.slice_label.setText("Slice 1/1")

    def _go_to_previous_slice(self) -> None:
        if self._slice_index > 0:
            self.slice_slider.setValue(self._slice_index - 1)

    def _go_to_next_slice(self) -> None:
        if self._slice_index < self._slice_count - 1:
            self.slice_slider.setValue(self._slice_index + 1)

    def _emit_slice_index_changed(self, value: int) -> None:
        self._slice_index = value
        self._slice_summary = self._format_slice_summary(self._slice_count, self._slice_index)
        self._sync_stack_controls()
        self._update_info_label()
        self.sliceIndexChanged.emit(value)

    def _set_stack_view_mode(self, mode: str) -> None:
        if self._stack_view_mode == mode:
            self._sync_stack_controls()
            return
        self._stack_view_mode = mode
        self._slice_summary = self._format_slice_summary(self._slice_count, self._slice_index)
        self._sync_stack_controls()
        self._update_info_label()
        self.stackViewModeChanged.emit(mode)


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(520, 420)

        self.checkpoint_edit = QLineEdit(str(settings.checkpoint_path))
        self.checkpoint_browse_btn = QPushButton("Browse...")

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(settings.threshold)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 10_000)
        self.batch_size_spin.setValue(settings.batch_size)

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(0, 10_000_000)
        self.min_area_spin.setValue(settings.min_area)

        self.resize_size_spin = QSpinBox()
        self.resize_size_spin.setRange(32, 4096)
        self.resize_size_spin.setSingleStep(32)
        self.resize_size_spin.setValue(settings.resize_size)

        self.background_radius_spin = QSpinBox()
        self.background_radius_spin.setRange(0, 10_000)
        self.background_radius_spin.setValue(settings.background_radius)

        self.px_per_micron_spin = QDoubleSpinBox()
        self.px_per_micron_spin.setRange(0.0, 10000.0)
        self.px_per_micron_spin.setDecimals(4)
        self.px_per_micron_spin.setSingleStep(0.1)
        self.px_per_micron_spin.setValue(settings.px_per_micron)

        self.overlay_alpha_spin = QDoubleSpinBox()
        self.overlay_alpha_spin.setRange(0.0, 1.0)
        self.overlay_alpha_spin.setDecimals(2)
        self.overlay_alpha_spin.setSingleStep(0.05)
        self.overlay_alpha_spin.setValue(settings.overlay_alpha)

        self.tiff_mode_combo = QComboBox()
        self.tiff_mode_combo.addItem("Current slice", TIFF_MODE_CURRENT_SLICE)
        self.tiff_mode_combo.addItem("Max projection", TIFF_MODE_MAX_PROJECTION)
        self.tiff_mode_combo.addItem("All slices", TIFF_MODE_ALL_SLICES)
        current_tiff_index = self.tiff_mode_combo.findData(settings.tiff_stack_mode)
        if current_tiff_index >= 0:
            self.tiff_mode_combo.setCurrentIndex(current_tiff_index)

        self.save_overlays_check = QCheckBox("Save overlays")
        self.save_overlays_check.setChecked(settings.save_overlays)
        self.save_masks_check = QCheckBox("Save masks")
        self.save_masks_check.setChecked(settings.save_masks)
        self.automatic_quant_check = QCheckBox("Run quantification automatically")
        self.automatic_quant_check.setChecked(settings.automatic_quantification)
        self.excel_check = QCheckBox("Generate Excel workbook")
        self.excel_check.setChecked(settings.excel_enabled)

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        form = QFormLayout()
        checkpoint_row = QWidget()
        checkpoint_layout = QHBoxLayout(checkpoint_row)
        checkpoint_layout.setContentsMargins(0, 0, 0, 0)
        checkpoint_layout.addWidget(self.checkpoint_edit, 1)
        checkpoint_layout.addWidget(self.checkpoint_browse_btn)

        form.addRow("Checkpoint path", checkpoint_row)
        form.addRow("Batch size", self.batch_size_spin)
        form.addRow("Probability threshold", self.threshold_spin)
        form.addRow("Minimum area", self.min_area_spin)
        form.addRow("Resize size", self.resize_size_spin)
        form.addRow("Background radius", self.background_radius_spin)
        form.addRow("Pixels per micron", self.px_per_micron_spin)
        form.addRow("Overlay alpha", self.overlay_alpha_spin)
        form.addRow("TIFF stack mode", self.tiff_mode_combo)

        toggles = QWidget()
        toggles_layout = QVBoxLayout(toggles)
        toggles_layout.setContentsMargins(0, 0, 0, 0)
        toggles_layout.addWidget(self.save_overlays_check)
        toggles_layout.addWidget(self.save_masks_check)
        toggles_layout.addWidget(self.automatic_quant_check)
        toggles_layout.addWidget(self.excel_check)
        toggles_layout.addStretch(1)
        form.addRow("Output options", toggles)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        root.addLayout(form)
        root.addStretch(1)
        root.addWidget(buttons)

    def _connect_signals(self) -> None:
        self.checkpoint_browse_btn.clicked.connect(self._browse_checkpoint)

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select checkpoint", self.checkpoint_edit.text())
        if path:
            self.checkpoint_edit.setText(path)

    def to_settings(self, current_settings: AppSettings) -> AppSettings:
        return AppSettings(
            checkpoint_path=Path(self.checkpoint_edit.text().strip()),
            batch_size=int(self.batch_size_spin.value()),
            threshold=float(self.threshold_spin.value()),
            min_area=int(self.min_area_spin.value()),
            background_radius=int(self.background_radius_spin.value()),
            resize_size=int(self.resize_size_spin.value()),
            px_per_micron=float(self.px_per_micron_spin.value()),
            overlay_alpha=float(self.overlay_alpha_spin.value()),
            save_overlays=self.save_overlays_check.isChecked(),
            save_masks=self.save_masks_check.isChecked(),
            automatic_quantification=self.automatic_quant_check.isChecked(),
            excel_enabled=self.excel_check.isChecked(),
            histogram_enabled=current_settings.histogram_enabled,
            tiff_stack_mode=str(self.tiff_mode_combo.currentData()),
        )
