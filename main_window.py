



from __future__ import annotations

import shlex
from pathlib import Path

from PySide6.QtCore import QUrl, Qt, Slot
from PySide6.QtGui import QAction, QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from backend import prepare_run
from config import DEFAULT_CHECKPOINT, DEFAULT_OUT_DIR, VALID_EXTS
from imaging import image_file_to_qpixmap, load_preview_bundle, numpy_image_to_qpixmap
from app_models import BatchRunResult
from widgets import ImagePanel
from workers import BatchProcessWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("UNetDC Segmenter")
        self.resize(1450, 900)

        self.worker: BatchProcessWorker | None = None
        self.last_out_dir: Path | None = None
        self._overlay_paths: list[Path] = []
        self._input_images: list[Path] = []
        self.folder_path = ""
        self.out_dir_path = DEFAULT_OUT_DIR

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 10000)
        self.batch_spin.setValue(8)
        self.batch_spin.setToolTip("Number of images processed together in one batch.")

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setValue(0.3)
        self.threshold_spin.setToolTip("Detection confidence threshold applied to predicted regions.")

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(0, 10000000)
        self.min_area_spin.setValue(1)
        self.min_area_spin.setToolTip("Discard detections smaller than this area in pixels.")

        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(0, 10000)
        self.radius_spin.setValue(50)
        self.radius_spin.setToolTip("Reference radius used for downstream quantification.")

        self.px_per_micron_spin = QDoubleSpinBox()
        self.px_per_micron_spin.setRange(0.0, 10000.0)
        self.px_per_micron_spin.setDecimals(4)
        self.px_per_micron_spin.setSingleStep(0.1)
        self.px_per_micron_spin.setValue(0.0)
        self.px_per_micron_spin.setToolTip("Calibration value used to convert pixels to microns.")

        self.save_overlays_check = QCheckBox("Save overlays")
        self.save_overlays_check.setChecked(True)
        self.excel_check = QCheckBox("Generate Excel workbook")
        self.excel_check.setChecked(True)

        self.open_folder_btn = QPushButton("Browse...")
        self.browse_out_btn = QPushButton("Browse...")
        self.clear_log_btn = QPushButton("Clear log")
        self.clear_overlays_btn = QPushButton("Clear overlays")
        self.run_btn = QPushButton("Run")
        self.run_btn.setDefault(True)
        self.run_btn.setMinimumHeight(40)
        self.open_output_btn = QPushButton("Open results")
        self.open_output_btn.setEnabled(False)
        self.open_output_btn.setMinimumHeight(36)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)

        self.status_label = QLabel("Select an input folder and output location, then run the pipeline.")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.original_panel = ImagePanel("Original preview")
        self.mask_panel = ImagePanel("Segmentation result")
        self.overlay_panel = ImagePanel("Overlay")
        self.input_list = QListWidget()
        self.input_list.setMinimumWidth(220)
        self.input_list.setEnabled(False)
        self.prev_image_btn = QPushButton("Previous")
        self.next_image_btn = QPushButton("Next")
        self.prev_image_btn.setEnabled(False)
        self.next_image_btn.setEnabled(False)

        self.summary_table = QTableWidget()
        self.stats_table = QTableWidget()
        self.droplets_table = QTableWidget()
        for table in (self.summary_table, self.stats_table, self.droplets_table):
            table.horizontalHeader().setStretchLastSection(True)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        self.summary_message = QLabel("Run the pipeline to see summary tables.")
        self.summary_message.setAlignment(Qt.AlignCenter)
        self.summary_message.setWordWrap(True)

        self.overlay_list = QListWidget()
        self.overlay_list.setMinimumWidth(180)
        self.overlay_list.setEnabled(False)
        self.overlay_image_label = QLabel("Overlay gallery will appear here after a successful run.")
        self.overlay_image_label.setAlignment(Qt.AlignCenter)
        self.overlay_image_label.setWordWrap(True)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)

        self.tabs = QTabWidget()

        self._build_ui()
        self._connect_signals()
        self._build_menu()

    def _build_ui(self) -> None:
        controls_group = QGroupBox("Controls")
        controls_layout = QGridLayout()
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(8)
        controls_layout.addWidget(QLabel("Folder"), 0, 0)
        controls_layout.addWidget(self.open_folder_btn, 0, 1)
        controls_layout.addWidget(QLabel("Output"), 0, 2)
        controls_layout.addWidget(self.browse_out_btn, 0, 3)

        controls_layout.addWidget(QLabel("Batch"), 1, 0)
        controls_layout.addWidget(self.batch_spin, 1, 1)
        controls_layout.addWidget(QLabel("Threshold"), 1, 2)
        controls_layout.addWidget(self.threshold_spin, 1, 3)
        controls_layout.addWidget(QLabel("Min area"), 1, 4)
        controls_layout.addWidget(self.min_area_spin, 1, 5)
        controls_layout.addWidget(QLabel("Radius"), 1, 6)
        controls_layout.addWidget(self.radius_spin, 1, 7)
        controls_layout.addWidget(QLabel("Pixel size (px/um)"), 1, 8)
        controls_layout.addWidget(self.px_per_micron_spin, 1, 9)
        controls_layout.addWidget(self.save_overlays_check, 1, 10)
        controls_layout.addWidget(self.excel_check, 1, 11)
        controls_layout.addWidget(self.clear_log_btn, 1, 12)
        controls_layout.addWidget(self.clear_overlays_btn, 1, 13)
        controls_layout.addWidget(self.progress_bar, 1, 14)
        controls_layout.addWidget(self.run_btn, 1, 15)
        controls_layout.addWidget(self.open_output_btn, 1, 16)

        controls_layout.addWidget(self.status_label, 2, 0, 1, 17)
        controls_layout.setColumnStretch(9, 2)
        controls_layout.setColumnStretch(14, 1)
        controls_group.setLayout(controls_layout)

        browser_widget = QWidget()
        browser_layout = QVBoxLayout(browser_widget)
        browser_layout.setContentsMargins(0, 0, 0, 0)
        browser_layout.addWidget(QLabel("Input images"))
        browser_layout.addWidget(self.input_list)
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_image_btn)
        nav_layout.addWidget(self.next_image_btn)
        browser_layout.addLayout(nav_layout)

        image_splitter = QSplitter(Qt.Horizontal)
        image_splitter.setChildrenCollapsible(False)
        image_splitter.addWidget(self.original_panel)
        image_splitter.addWidget(self.mask_panel)
        image_splitter.addWidget(self.overlay_panel)
        image_splitter.setStretchFactor(0, 1)
        image_splitter.setStretchFactor(1, 1)
        image_splitter.setStretchFactor(2, 1)
        image_splitter.setSizes([120, 120, 120])

        preview_splitter = QSplitter(Qt.Horizontal)
        preview_splitter.setChildrenCollapsible(False)
        preview_splitter.addWidget(browser_widget)
        preview_splitter.addWidget(image_splitter)
        preview_splitter.setStretchFactor(0, 0)
        preview_splitter.setStretchFactor(1, 1)
        preview_splitter.setSizes([260, 1180])

        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.addWidget(preview_splitter)
        preview_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.summary_tab)
        summary_layout.addWidget(self.summary_message)
        summary_layout.addWidget(QLabel("Per-image summary"))
        summary_layout.addWidget(self.summary_table)
        summary_layout.addWidget(QLabel("Size statistics"))
        summary_layout.addWidget(self.stats_table)
        self.tabs.addTab(self.summary_tab, "Summary")

        droplets_widget = QWidget()
        droplets_layout = QVBoxLayout(droplets_widget)
        droplets_layout.addWidget(QLabel("Per-droplet table"))
        droplets_layout.addWidget(self.droplets_table)
        self.tabs.addTab(droplets_widget, "Droplets")

        overlay_widget = QWidget()
        overlay_layout = QHBoxLayout(overlay_widget)
        overlay_layout.addWidget(self.overlay_list)
        overlay_scroll = QScrollArea()
        overlay_scroll.setWidgetResizable(True)
        overlay_image_container = QWidget()
        overlay_image_layout = QVBoxLayout(overlay_image_container)
        overlay_image_layout.addWidget(self.overlay_image_label)
        overlay_scroll.setWidget(overlay_image_container)
        overlay_layout.addWidget(overlay_scroll, 1)
        self.tabs.addTab(overlay_widget, "Overlays")

        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.addWidget(self.log_output)
        self.tabs.addTab(log_widget, "Log")

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.addWidget(controls_group)
        main_splitter.addWidget(preview_group)
        main_splitter.addWidget(self.tabs)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 4)
        main_splitter.setStretchFactor(2, 2)
        main_splitter.setSizes([60, 520, 220])

        root.addWidget(main_splitter)
        self.setCentralWidget(central)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_folder_action = QAction("Open folder", self)
        open_folder_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_folder_action)

        run_action = QAction("Run", self)
        run_action.triggered.connect(self.run_pipeline)
        file_menu.addAction(run_action)

        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _connect_signals(self) -> None:
        self.open_folder_btn.clicked.connect(self.open_folder)
        self.browse_out_btn.clicked.connect(self.browse_output_dir)
        self.clear_log_btn.clicked.connect(self.clear_log)
        self.clear_overlays_btn.clicked.connect(self.clear_overlays)
        self.run_btn.clicked.connect(self.run_pipeline)
        self.open_output_btn.clicked.connect(self.open_output_folder)
        self.overlay_list.currentRowChanged.connect(self.on_overlay_selected)
        self.input_list.currentRowChanged.connect(self.on_input_selected)
        self.prev_image_btn.clicked.connect(self.show_previous_image)
        self.next_image_btn.clicked.connect(self.show_next_image)

    def open_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not path:
            return
        self.folder_path = path
        folder = Path(path)
        self._clear_tables_and_outputs()
        self._input_images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS)
        self._populate_input_list()
        self._refresh_status_label()
        if not self._input_images:
            self._append_log_line(f"Selected folder: {path}")
            self._append_log_line("Detected images: 0")
            self.original_panel.clear_to_title("Original preview")
            return

        self.input_list.setCurrentRow(0)
        self._append_log_line(f"Selected folder: {path}")
        self._append_log_line(f"Detected images: {len(self._input_images)}")

    def browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.out_dir_path = path
            self._refresh_status_label()

    def clear_log(self) -> None:
        self.log_output.clear()

    def clear_overlays(self) -> None:
        self.overlay_list.clear()
        self.overlay_list.setEnabled(False)
        self.overlay_image_label.setPixmap(QPixmap())
        self.overlay_image_label.setText("Overlay gallery cleared.")
        self._overlay_paths = []
        self.overlay_panel.clear_to_title("Overlay")

    def run_pipeline(self) -> None:
        if self.worker is not None:
            return

        try:
            input_path, command, out_dir = prepare_run(
                folder_text=self.folder_path,
                ckpt_text=DEFAULT_CHECKPOINT,
                out_dir_text=self.out_dir_path,
                batch_value=int(self.batch_spin.value()),
                threshold_value=float(self.threshold_spin.value()),
                min_area_value=int(self.min_area_spin.value()),
                radius_value=int(self.radius_spin.value()),
                px_per_micron_value=float(self.px_per_micron_spin.value()),
                save_overlays=self.save_overlays_check.isChecked(),
                excel_enabled=self.excel_check.isChecked(),
                histogram_enabled=False,
            )
        except ValueError as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return

        self.log_output.clear()
        self._append_log_line("Running: " + " ".join(shlex.quote(a) for a in command))
        self._set_running(True)
        self._clear_tables_and_outputs()
        self.last_out_dir = out_dir

        self.worker = BatchProcessWorker(
            input_path=input_path,
            command=command,
            out_dir=out_dir,
            parent=self,
        )
        self.worker.output.connect(self._append_log_line)
        self.worker.succeeded.connect(self.on_run_succeeded)
        self.worker.failed.connect(self.on_run_failed)
        self.worker.finished.connect(self._cleanup_worker)
        self.worker.start()

    def _set_running(self, running: bool) -> None:
        widgets = (
            self.batch_spin,
            self.threshold_spin,
            self.min_area_spin,
            self.radius_spin,
            self.px_per_micron_spin,
            self.save_overlays_check,
            self.excel_check,
            self.open_folder_btn,
            self.browse_out_btn,
        )
        for widget in widgets:
            widget.setEnabled(not running)
        if running:
            self.input_list.setEnabled(False)
            self.prev_image_btn.setEnabled(False)
            self.next_image_btn.setEnabled(False)
            self.status_label.setText("Processing images. Progress details will appear in the log tab.")
        else:
            self.input_list.setEnabled(bool(self._input_images))
            self._update_navigation_buttons()
            if self.last_out_dir is not None and self.last_out_dir.exists():
                self.status_label.setText(f"Ready. Latest results are available in:\n{self.last_out_dir}")
            else:
                self.status_label.setText("Select an input folder and output location, then run the pipeline.")
        self.run_btn.setEnabled(not running)
        self.progress_bar.setVisible(running)
        self.open_output_btn.setEnabled((not running) and self.last_out_dir is not None and self.last_out_dir.exists())

    @Slot()
    def _cleanup_worker(self) -> None:
        self._set_running(False)
        self.worker = None

    @Slot(str)
    def _append_log_line(self, line: str) -> None:
        self.log_output.appendPlainText(line)

    @Slot(object)
    def on_run_succeeded(self, result: BatchRunResult) -> None:
        self._append_log_line("Finished successfully.")
        self._populate_summary(result.summary_rows, result.stats_rows)
        self._populate_droplets(result.droplet_rows)
        self._load_overlay_gallery(Path(result.out_dir))
        self._load_result_previews(Path(result.out_dir))
        self.status_label.setText(f"Processing complete. Results saved to:\n{result.out_dir}")

        self.open_output_btn.setEnabled(True)
        self.tabs.setCurrentWidget(self.summary_tab)
        self.raise_()
        self.activateWindow()
        QMessageBox.information(self, "Done", "Processing complete.")

    @Slot(str)
    def on_run_failed(self, message: str) -> None:
        self.log_output.appendPlainText("ERROR: " + message)
        self.status_label.setText("Run failed. Check the log tab for details and adjust the settings if needed.")
        QMessageBox.critical(self, "Error", message)

    def _clear_tables_and_outputs(self) -> None:
        self._reset_table(self.summary_table)
        self._reset_table(self.stats_table)
        self._reset_table(self.droplets_table)
        self.summary_message.setText("Run the pipeline to see summary tables.")
        self.overlay_list.clear()
        self.overlay_list.setEnabled(False)
        self.overlay_image_label.setPixmap(QPixmap())
        self.overlay_image_label.setText("Overlay gallery will appear here after a successful run.")
        self._overlay_paths = []
        self.mask_panel.clear_to_title("Segmentation result")
        self.overlay_panel.clear_to_title("Overlay")
        self._update_navigation_buttons()

    def _populate_summary(self, summary_rows: list[dict[str, str]], stats_rows: list[dict[str, str]]) -> None:
        if summary_rows:
            headers = list(summary_rows[0].keys())
            self._populate_table(self.summary_table, headers, summary_rows)
            self.summary_message.setText("")
        else:
            self.summary_message.setText("Summary files were not generated.")

        if stats_rows:
            headers = list(stats_rows[0].keys())
            self._populate_table(self.stats_table, headers, stats_rows)

    def _populate_droplets(self, droplet_rows: list[dict[str, str]]) -> None:
        if not droplet_rows:
            self._reset_table(self.droplets_table)
            return
        headers = list(droplet_rows[0].keys())
        self._populate_table(self.droplets_table, headers, droplet_rows)

    def _refresh_status_label(self) -> None:
        folder_text = self.folder_path if self.folder_path else "No folder selected"
        output_text = self.out_dir_path if self.out_dir_path else "No output folder selected"
        self.status_label.setText(f"Folder: {folder_text}\nOutput: {output_text}")

    def _load_overlay_gallery(self, out_dir: Path) -> None:
        overlay_dir = out_dir / "overlays"
        overlay_files = sorted([p for p in overlay_dir.glob("*") if p.suffix.lower() in VALID_EXTS]) if overlay_dir.exists() else []

        self.overlay_list.clear()
        self._overlay_paths = overlay_files
        if not self._overlay_paths:
            self.overlay_list.setEnabled(False)
            self.overlay_image_label.setPixmap(QPixmap())
            self.overlay_image_label.setText("No overlay images were generated.")
            return

        self.overlay_list.setEnabled(True)
        for path in self._overlay_paths:
            self.overlay_list.addItem(path.name)
        self.overlay_list.setCurrentRow(0)

    @Slot(int)
    def on_overlay_selected(self, index: int) -> None:
        if index < 0 or index >= len(self._overlay_paths):
            return
        path = self._overlay_paths[index]
        pixmap = image_file_to_qpixmap(path)
        if pixmap.isNull():
            self.overlay_image_label.setPixmap(QPixmap())
            self.overlay_image_label.setText(f"Could not load overlay: {path.name}")
            return
        self.overlay_image_label.setText("")
        self.overlay_image_label.setPixmap(pixmap)
        self.overlay_panel.set_scaled_pixmap(pixmap)
        self._show_related_images_for_overlay(path)

    @Slot(int)
    def on_input_selected(self, index: int) -> None:
        if index < 0 or index >= len(self._input_images):
            self.original_panel.clear_to_title("Original preview")
            self.mask_panel.clear_to_title("Segmentation result")
            self.overlay_panel.clear_to_title("Overlay")
            self._update_navigation_buttons()
            return

        current_path = self._input_images[index]
        self._show_input_preview(current_path)
        self._show_outputs_for_stem(current_path.stem)
        self._update_navigation_buttons()

    def show_previous_image(self) -> None:
        current = self.input_list.currentRow()
        if current > 0:
            self.input_list.setCurrentRow(current - 1)

    def show_next_image(self) -> None:
        current = self.input_list.currentRow()
        if current < len(self._input_images) - 1:
            self.input_list.setCurrentRow(current + 1)

    def _populate_input_list(self) -> None:
        self.input_list.blockSignals(True)
        self.input_list.clear()
        for path in self._input_images:
            self.input_list.addItem(path.name)
        self.input_list.setEnabled(bool(self._input_images))
        self.input_list.blockSignals(False)
        self._update_navigation_buttons()

    def _update_navigation_buttons(self) -> None:
        has_images = bool(self._input_images) and self.input_list.isEnabled()
        current = self.input_list.currentRow()
        self.prev_image_btn.setEnabled(has_images and current > 0)
        self.next_image_btn.setEnabled(has_images and 0 <= current < len(self._input_images) - 1)

    def _show_input_preview(self, path: Path) -> None:
        try:
            bundle = load_preview_bundle(str(path))
            pixmap = numpy_image_to_qpixmap(bundle.display_image)
        except Exception:
            pixmap = QPixmap()

        if pixmap.isNull():
            self.original_panel.clear_to_title(f"Could not load preview:\n{path.name}")
            return
        self.original_panel.set_scaled_pixmap(pixmap)

    def _load_result_previews(self, out_dir: Path) -> None:
        current_index = self.input_list.currentRow()
        if 0 <= current_index < len(self._input_images):
            self._show_outputs_for_stem(self._input_images[current_index].stem)
            return

        overlay_path = self._first_image_in_dir(out_dir / "overlays")
        if overlay_path is not None:
            self._show_related_images_for_overlay(overlay_path)
            return

        self.mask_panel.clear_to_title("Segmentation result")
        self.overlay_panel.clear_to_title("Overlay")

    def _show_related_images_for_overlay(self, overlay_path: Path) -> None:
        stem = overlay_path.stem.removesuffix("_overlay")
        input_match = self._find_input_image(stem)
        if input_match is not None:
            row = self._input_images.index(input_match)
            if self.input_list.currentRow() != row:
                self.input_list.setCurrentRow(row)
                return

        self._show_outputs_for_stem(stem)

    def _show_outputs_for_stem(self, stem: str) -> None:
        mask_path = self._output_image_path("predicted_masks", stem, "_pred.png")
        overlay_path = self._output_image_path("overlays", stem, "_overlay.png")

        if mask_path is None:
            self.mask_panel.clear_to_title("Segmentation result")
        else:
            mask_pixmap = image_file_to_qpixmap(mask_path)
            if mask_pixmap.isNull():
                self.mask_panel.clear_to_title(f"Could not load mask:\n{mask_path.name}")
            else:
                self.mask_panel.set_scaled_pixmap(mask_pixmap)

        if overlay_path is None:
            self.overlay_panel.clear_to_title("Overlay")
        else:
            overlay_pixmap = image_file_to_qpixmap(overlay_path)
            if overlay_pixmap.isNull():
                self.overlay_panel.clear_to_title(f"Could not load overlay:\n{overlay_path.name}")
            else:
                self.overlay_panel.set_scaled_pixmap(overlay_pixmap)
                self.overlay_image_label.setText("")
                self.overlay_image_label.setPixmap(overlay_pixmap)
            overlay_index = self._overlay_index_for_path(overlay_path)
            if overlay_index is not None and self.overlay_list.currentRow() != overlay_index:
                self.overlay_list.blockSignals(True)
                self.overlay_list.setCurrentRow(overlay_index)
                self.overlay_list.blockSignals(False)

    def _find_input_image(self, stem: str) -> Path | None:
        for path in self._input_images:
            if path.stem == stem:
                return path
        return None

    def _output_image_path(self, folder_name: str, stem: str, suffix: str) -> Path | None:
        if self.last_out_dir is None:
            return None

        candidate = self.last_out_dir / folder_name / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
        return None

    def _overlay_index_for_path(self, target: Path) -> int | None:
        for index, path in enumerate(self._overlay_paths):
            if path == target:
                return index
        return None

    def _first_image_in_dir(self, directory: Path) -> Path | None:
        if not directory.exists():
            return None

        images = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS)
        if not images:
            return None
        return images[0]

    def _reset_table(self, table: QTableWidget) -> None:
        table.clear()
        table.setRowCount(0)
        table.setColumnCount(0)

    def _populate_table(self, table: QTableWidget, headers: list[str], rows: list[dict[str, str]]) -> None:
        self._reset_table(table)
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for col_index, header in enumerate(headers):
                item = QTableWidgetItem(str(row.get(header, "")))
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                table.setItem(row_index, col_index, item)
        table.resizeColumnsToContents()

    def open_output_folder(self) -> None:
        if self.last_out_dir is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_out_dir)))
