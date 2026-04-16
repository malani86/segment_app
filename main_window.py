from __future__ import annotations

import shutil
import shlex
from pathlib import Path

from PySide6.QtCore import QUrl, Qt, Slot
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QFrame,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app_models import (
    BatchRunResult,
    INPUT_MODE_BATCH,
    INPUT_MODE_SINGLE,
    STACK_VIEW_PROJECTION,
    STACK_VIEW_SLICE,
    TIFF_MODE_ALL_SLICES,
    TIFF_MODE_CURRENT_SLICE,
    TIFF_MODE_MAX_PROJECTION,
    VIEWER_MODE_OVERLAY,
    WORKFLOW_STEP_EXPORT,
    WORKFLOW_STEP_LOAD,
    WORKFLOW_STEP_PREVIEW,
    WORKFLOW_STEP_QUANTIFY,
    WORKFLOW_STEP_SEGMENT,
)
from controller import SegmentAppController
from preview_service import PreviewResultService, ViewerDisplayData
from widgets import InspectionViewer, SettingsDialog
from workers import BatchProcessWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("UNetDC Segmenter")
        self.resize(1450, 900)

        self.controller = SegmentAppController()
        self.preview_service = PreviewResultService()
        self.worker: BatchProcessWorker | None = None
        self._pending_temp_input_dir: Path | None = None

        self.viewer = InspectionViewer()
        self.input_list = QListWidget()
        self.input_list.setMinimumWidth(220)
        self.input_list.setEnabled(False)

        self.summary_table = QTableWidget()
        self.stats_table = QTableWidget()
        self.droplets_table = QTableWidget()
        for table in (self.summary_table, self.stats_table, self.droplets_table):
            table.horizontalHeader().setStretchLastSection(True)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        self.summary_message = QLabel("Run the pipeline to see summary tables.")
        self.summary_message.setAlignment(Qt.AlignCenter)
        self.summary_message.setWordWrap(True)
        self.tiff_mode_label = QLabel()
        self.tiff_mode_label.setWordWrap(True)
        self.tiff_mode_label.setStyleSheet("color: #6b7280;")

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_clear_btn = QPushButton("Clear log")

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self._build_ui()
        self._connect_signals()
        self._build_menu()
        self._set_workflow_step(WORKFLOW_STEP_LOAD)
        self._refresh_tiff_mode_label()

    def _build_ui(self) -> None:
        sidebar_frame = QFrame()
        sidebar_frame.setObjectName("imageSidebar")
        sidebar_frame.setStyleSheet(
            "#imageSidebar { border: 1px solid #e5e7eb; border-radius: 12px; background: #fbfbfc; }"
        )
        sidebar_frame.setMinimumWidth(210)
        sidebar_frame.setMaximumWidth(260)
        sidebar_layout = QVBoxLayout(sidebar_frame)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        sidebar_layout.setSpacing(10)
        sidebar_title = QLabel("Images")
        sidebar_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #1f2937;")
        sidebar_hint = QLabel("Select an image to inspect and analyze.")
        sidebar_hint.setWordWrap(True)
        sidebar_hint.setStyleSheet("color: #6b7280;")
        sidebar_layout.addWidget(sidebar_title)
        sidebar_layout.addWidget(sidebar_hint)
        sidebar_layout.addWidget(self.input_list, 1)

        viewer_frame = QFrame()
        viewer_frame.setObjectName("viewerFrame")
        viewer_frame.setStyleSheet(
            "#viewerFrame { border: 1px solid #d7dbe2; border-radius: 12px; background: #ffffff; }"
        )
        viewer_layout = QVBoxLayout(viewer_frame)
        viewer_layout.setContentsMargins(12, 12, 12, 12)
        viewer_layout.setSpacing(10)
        viewer_title = QLabel("Image Viewer")
        viewer_title.setStyleSheet("font-size: 16px; font-weight: 700; color: #111827;")
        viewer_subtitle = QLabel("Inspect the original image, mask, and overlay in one place.")
        viewer_subtitle.setWordWrap(True)
        viewer_subtitle.setStyleSheet("color: #6b7280;")
        viewer_layout.addWidget(viewer_title)
        viewer_layout.addWidget(viewer_subtitle)
        viewer_layout.addWidget(self.tiff_mode_label)
        viewer_layout.addWidget(self.viewer, 1)

        results_frame = QFrame()
        results_frame.setObjectName("resultsFrame")
        results_frame.setStyleSheet(
            "#resultsFrame { border: 1px solid #d7dbe2; border-radius: 12px; background: #ffffff; }"
        )
        results_frame.setMinimumWidth(320)
        results_frame.setMaximumWidth(440)
        results_layout = QVBoxLayout(results_frame)
        results_layout.setContentsMargins(12, 12, 12, 12)
        results_layout.setSpacing(10)
        results_title = QLabel("Results")
        results_title.setStyleSheet("font-size: 16px; font-weight: 700; color: #111827;")
        results_subtitle = QLabel("Review summary tables and processing logs.")
        results_subtitle.setWordWrap(True)
        results_subtitle.setStyleSheet("color: #6b7280;")
        results_layout.addWidget(results_title)
        results_layout.addWidget(results_subtitle)
        results_layout.addWidget(self.tabs, 1)

        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.summary_tab)
        summary_layout.addWidget(self.summary_message)
        summary_layout.addWidget(QLabel("Per-image summary"))
        summary_layout.addWidget(self.summary_table)
        summary_layout.addWidget(QLabel("Size statistics"))
        summary_layout.addWidget(self.stats_table)
        self.tabs.addTab(self.summary_tab, "Summary")

        self.droplets_tab = QWidget()
        droplets_layout = QVBoxLayout(self.droplets_tab)
        droplets_layout.addWidget(QLabel("Per-droplet table"))
        droplets_layout.addWidget(self.droplets_table)
        self.tabs.addTab(self.droplets_tab, "Droplets")

        self.log_tab = QWidget()
        log_layout = QVBoxLayout(self.log_tab)
        log_toolbar = QHBoxLayout()
        log_toolbar.addStretch(1)
        log_toolbar.addWidget(self.log_clear_btn)
        log_layout.addLayout(log_toolbar)
        log_layout.addWidget(self.log_output)
        self.tabs.addTab(self.log_tab, "Log")

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        workspace_splitter = QSplitter(Qt.Horizontal)
        workspace_splitter.setChildrenCollapsible(False)
        workspace_splitter.addWidget(sidebar_frame)
        workspace_splitter.addWidget(viewer_frame)
        workspace_splitter.addWidget(results_frame)
        workspace_splitter.setStretchFactor(0, 0)
        workspace_splitter.setStretchFactor(1, 1)
        workspace_splitter.setStretchFactor(2, 0)
        workspace_splitter.setSizes([220, 980, 420])

        root.addWidget(workspace_splitter, 1)
        self.setCentralWidget(central)

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()
        menu_bar.clear()
        file_menu = menu_bar.addMenu("File")

        self.open_folder_action = QAction("Open Folder", self)
        self.open_folder_action.triggered.connect(self.open_folder)
        file_menu.addAction(self.open_folder_action)

        self.open_image_action = QAction("Open Image", self)
        self.open_image_action.triggered.connect(self.open_image)
        file_menu.addAction(self.open_image_action)

        file_menu.addSeparator()

        self.select_output_action = QAction("Select Output Folder", self)
        self.select_output_action.triggered.connect(self.browse_output_dir)
        file_menu.addAction(self.select_output_action)

        self.open_output_action = QAction("Open Output Folder", self)
        self.open_output_action.triggered.connect(self.open_output_folder)
        self.open_output_action.setEnabled(False)
        file_menu.addAction(self.open_output_action)

        file_menu.addSeparator()

        self.exit_action = QAction("Exit", self)
        self.exit_action.triggered.connect(self.close)
        file_menu.addAction(self.exit_action)

        run_menu = menu_bar.addMenu("Run")

        self.run_action = QAction("Run Analysis", self)
        self.run_action.triggered.connect(self.run_pipeline)
        run_menu.addAction(self.run_action)

        navigate_menu = menu_bar.addMenu("Navigate")

        self.previous_image_action = QAction("Previous Image", self)
        self.previous_image_action.triggered.connect(self.show_previous_image)
        self.previous_image_action.setEnabled(False)
        navigate_menu.addAction(self.previous_image_action)

        self.next_image_action = QAction("Next Image", self)
        self.next_image_action.triggered.connect(self.show_next_image)
        self.next_image_action.setEnabled(False)
        navigate_menu.addAction(self.next_image_action)

        tools_menu = menu_bar.addMenu("Tools")

        self.settings_action = QAction("Settings", self)
        self.settings_action.triggered.connect(self.open_settings_dialog)
        tools_menu.addAction(self.settings_action)

        help_menu = menu_bar.addMenu("Help")

        self.about_action = QAction("About", self)
        self.about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(self.about_action)

    def _connect_signals(self) -> None:
        self.log_clear_btn.clicked.connect(self.clear_log)
        self.input_list.currentRowChanged.connect(self.on_input_selected)
        self.viewer.modeChanged.connect(self._on_viewer_mode_changed)
        self.viewer.fitModeChanged.connect(self._on_viewer_fit_mode_changed)
        self.viewer.sliceIndexChanged.connect(self._on_viewer_slice_index_changed)
        self.viewer.stackViewModeChanged.connect(self._on_viewer_stack_view_mode_changed)
        self.tabs.currentChanged.connect(self._on_tab_changed)

    @property
    def state(self):
        return self.controller.state

    def _set_workflow_step(self, step_key: str) -> None:
        self.state.viewer.workflow_step = step_key

    @Slot(str)
    def _on_viewer_mode_changed(self, mode: str) -> None:
        self.state.viewer.current_mode = mode
        self._set_workflow_step(WORKFLOW_STEP_PREVIEW)

    @Slot(bool)
    def _on_viewer_fit_mode_changed(self, enabled: bool) -> None:
        self.state.viewer.fit_to_window = enabled

    @Slot(int)
    def _on_viewer_slice_index_changed(self, index: int) -> None:
        self.state.viewer.current_slice_index = index
        self._refresh_tiff_mode_label()
        self._refresh_viewer()

    @Slot(str)
    def _on_viewer_stack_view_mode_changed(self, mode: str) -> None:
        self.state.viewer.stack_view_mode = mode
        self._refresh_viewer()

    @Slot(int)
    def _on_tab_changed(self, index: int) -> None:
        if index in (0, 1):
            self._set_workflow_step(WORKFLOW_STEP_QUANTIFY)
        elif index == 2:
            self._set_workflow_step(WORKFLOW_STEP_SEGMENT)

    def _sync_settings_to_state(self) -> None:
        self.controller.update_settings(
            checkpoint_path=self.state.settings.checkpoint_path,
            batch_size=int(self.state.settings.batch_size),
            threshold=float(self.state.settings.threshold),
            min_area=int(self.state.settings.min_area),
            background_radius=int(self.state.settings.background_radius),
            resize_size=int(self.state.settings.resize_size),
            px_per_micron=float(self.state.settings.px_per_micron),
            overlay_alpha=float(self.state.settings.overlay_alpha),
            save_overlays=self.state.settings.save_overlays,
            save_masks=self.state.settings.save_masks,
            automatic_quantification=self.state.settings.automatic_quantification,
            excel_enabled=self.state.settings.excel_enabled,
            histogram_enabled=self.state.settings.histogram_enabled,
            tiff_stack_mode=self.state.settings.tiff_stack_mode,
        )

    def open_settings_dialog(self) -> None:
        dialog = SettingsDialog(self.state.settings, self)
        if not dialog.exec():
            return

        updated_settings = dialog.to_settings(self.state.settings)
        self.controller.update_settings(
            checkpoint_path=updated_settings.checkpoint_path,
            batch_size=updated_settings.batch_size,
            threshold=updated_settings.threshold,
            min_area=updated_settings.min_area,
            background_radius=updated_settings.background_radius,
            resize_size=updated_settings.resize_size,
            px_per_micron=updated_settings.px_per_micron,
            overlay_alpha=updated_settings.overlay_alpha,
            save_overlays=updated_settings.save_overlays,
            save_masks=updated_settings.save_masks,
            automatic_quantification=updated_settings.automatic_quantification,
            excel_enabled=updated_settings.excel_enabled,
            histogram_enabled=updated_settings.histogram_enabled,
            tiff_stack_mode=updated_settings.tiff_stack_mode,
        )
        self._refresh_tiff_mode_label()
        self._append_log_line("Settings updated.")

    def _describe_tiff_mode(self) -> str:
        mode = self.state.settings.tiff_stack_mode
        if mode == TIFF_MODE_MAX_PROJECTION:
            return "TIFF inference mode: max projection for multi-slice TIFF stacks."
        if mode == TIFF_MODE_ALL_SLICES:
            return "TIFF inference mode: all slices for multi-slice TIFF stacks."
        return f"TIFF inference mode: current slice ({self.state.viewer.current_slice_index + 1}) for multi-slice TIFF stacks."

    def _refresh_tiff_mode_label(self) -> None:
        suffix = ""
        if self.state.viewer.is_stack:
            suffix = f" Viewing {self.state.viewer.available_slices} slice(s) in the selected stack."
        self.tiff_mode_label.setText(self._describe_tiff_mode() + suffix)

    def show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            "About segment_app",
            "segment_app\n\nA microscopy image analysis GUI for previewing, segmenting, and reviewing results.",
        )

    def open_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not path:
            return
        self._clear_tables_and_outputs()
        self.controller.set_input_dir(path)
        self._populate_input_list()
        self._set_workflow_step(WORKFLOW_STEP_LOAD)
        self._refresh_tiff_mode_label()
        if not self.state.session.input_images:
            self._append_log_line(f"Selected folder: {path}")
            self._append_log_line("Detected images: 0")
            self.viewer.clear("Select an image to inspect.")
            return

        self.input_list.setCurrentRow(0)
        self._append_log_line(f"Selected folder: {path}")
        self._append_log_line(f"Detected images: {len(self.state.session.input_images)}")

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff)",
        )
        if not path:
            return
        self._clear_tables_and_outputs()
        self.controller.set_input_file(path)
        self._populate_input_list()
        self._set_workflow_step(WORKFLOW_STEP_LOAD)
        self._refresh_tiff_mode_label()
        self.input_list.setCurrentRow(0)
        self._append_log_line(f"Selected image: {path}")

    def browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.controller.set_output_dir(path)

    def clear_log(self) -> None:
        self.log_output.clear()

    def run_pipeline(self) -> None:
        if self.worker is not None:
            return

        self._sync_settings_to_state()
        try:
            run_request = self.controller.build_run_request()
        except ValueError as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return

        self.log_output.clear()
        self._append_log_line("Running: " + " ".join(shlex.quote(a) for a in run_request.command))
        self._set_workflow_step(WORKFLOW_STEP_SEGMENT)
        self._set_running(True)
        self._clear_tables_and_outputs()
        self.controller.set_last_out_dir(run_request.out_dir)
        self._pending_temp_input_dir = run_request.temp_input_dir

        self.worker = BatchProcessWorker(
            input_path=run_request.input_path,
            command=run_request.command,
            out_dir=run_request.out_dir,
            parent=self,
        )
        self.worker.output.connect(self._append_log_line)
        self.worker.succeeded.connect(self.on_run_succeeded)
        self.worker.failed.connect(self.on_run_failed)
        self.worker.finished.connect(self._cleanup_worker)
        self.worker.start()

    def _set_running(self, running: bool) -> None:
        self.state.session.is_running = running
        widgets = (
            self.open_folder_action,
            self.open_image_action,
            self.select_output_action,
            self.settings_action,
        )
        for widget in widgets:
            widget.setEnabled(not running)
        self.run_action.setEnabled(not running)
        if running:
            self.input_list.setEnabled(False)
            self.previous_image_action.setEnabled(False)
            self.next_image_action.setEnabled(False)
        else:
            self.open_folder_action.setEnabled(True)
            self.open_image_action.setEnabled(True)
            self.input_list.setEnabled(bool(self.state.session.input_images))
            self._update_navigation_buttons()
        self.open_output_action.setEnabled(
            (not running)
            and self.state.session.last_out_dir is not None
            and self.state.session.last_out_dir.exists()
        )

    @Slot()
    def _cleanup_worker(self) -> None:
        self._set_running(False)
        self.worker = None
        if self._pending_temp_input_dir is not None:
            shutil.rmtree(self._pending_temp_input_dir, ignore_errors=True)
            self._pending_temp_input_dir = None

    @Slot(str)
    def _append_log_line(self, line: str) -> None:
        self.log_output.appendPlainText(line)

    @Slot(object)
    def on_run_succeeded(self, result: BatchRunResult) -> None:
        self.state.session.last_result = result
        self.controller.set_last_out_dir(Path(result.out_dir))
        self._append_log_line("Finished successfully.")
        self._populate_summary(result.summary_rows, result.stats_rows)
        self._populate_droplets(result.droplet_rows)
        self._load_result_images(Path(result.out_dir))
        self._load_result_previews()
        self._set_workflow_step(WORKFLOW_STEP_QUANTIFY)

        self.open_output_action.setEnabled(True)
        self.tabs.setCurrentWidget(self.summary_tab)
        self.raise_()
        self.activateWindow()
        QMessageBox.information(self, "Done", "Processing complete.")

    @Slot(str)
    def on_run_failed(self, message: str) -> None:
        self.log_output.appendPlainText("ERROR: " + message)
        self._set_workflow_step(WORKFLOW_STEP_SEGMENT)
        QMessageBox.critical(self, "Error", message)

    def _clear_tables_and_outputs(self) -> None:
        self.controller.clear_results()
        self._reset_table(self.summary_table)
        self._reset_table(self.stats_table)
        self._reset_table(self.droplets_table)
        self.summary_message.setText("Run the pipeline to see summary tables.")
        self.controller.set_overlay_paths([])
        self.preview_service.clear_preview_cache()
        self.viewer.clear("Select an image to inspect.")
        self._set_workflow_step(WORKFLOW_STEP_LOAD)
        self._update_navigation_buttons()
        self._refresh_tiff_mode_label()

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

    def _load_result_images(self, out_dir: Path) -> None:
        overlay_paths = self.preview_service.load_overlay_paths(out_dir)
        self.controller.set_overlay_paths(overlay_paths)

    @Slot(int)
    def on_input_selected(self, index: int) -> None:
        self.state.viewer.current_input_index = index
        if index < 0 or index >= len(self.state.session.input_images):
            self.viewer.clear("Select an image to inspect.")
            self._update_navigation_buttons()
            return

        self.state.viewer.current_slice_index = 0
        self.state.viewer.stack_view_mode = STACK_VIEW_SLICE
        self.preview_service.clear_preview_cache()
        self._refresh_viewer()
        self._set_workflow_step(WORKFLOW_STEP_PREVIEW)
        self._update_navigation_buttons()
        self._refresh_tiff_mode_label()

    def show_previous_image(self) -> None:
        current = self.input_list.currentRow()
        if current > 0:
            self.input_list.setCurrentRow(current - 1)

    def show_next_image(self) -> None:
        current = self.input_list.currentRow()
        if current < len(self.state.session.input_images) - 1:
            self.input_list.setCurrentRow(current + 1)

    def _populate_input_list(self) -> None:
        self.input_list.blockSignals(True)
        self.input_list.clear()
        for path in self.state.session.input_images:
            self.input_list.addItem(path.name)
        self.input_list.setEnabled(bool(self.state.session.input_images))
        self.input_list.blockSignals(False)
        self._update_navigation_buttons()

    def _update_navigation_buttons(self) -> None:
        has_images = bool(self.state.session.input_images) and self.input_list.isEnabled()
        current = self.input_list.currentRow()
        self.previous_image_action.setEnabled(has_images and current > 0)
        self.next_image_action.setEnabled(has_images and 0 <= current < len(self.state.session.input_images) - 1)

    def _load_result_previews(self) -> None:
        current_index = self.input_list.currentRow()
        if 0 <= current_index < len(self.state.session.input_images):
            if self.state.settings.tiff_stack_mode == TIFF_MODE_MAX_PROJECTION:
                self.state.viewer.stack_view_mode = STACK_VIEW_PROJECTION
            else:
                self.state.viewer.stack_view_mode = STACK_VIEW_SLICE
            self._refresh_viewer()
            return

        overlay_path = self.preview_service.first_overlay_path_from_paths(self.state.session.overlay_paths)
        if overlay_path is not None:
            result_stem = overlay_path.stem.removesuffix("_overlay")
            selection = self.preview_service.get_preview_bundle(
                viewer_state=self.state.viewer,
                input_images=self.state.session.input_images,
                stem=result_stem,
                last_out_dir=self.state.session.last_out_dir,
            )
            matched_index = self.preview_service.find_input_index_by_stem(
                selection.normalized_stem,
                self.state.session.input_images,
            )
            if matched_index is not None:
                if selection.matched_slice_index is not None:
                    self.state.viewer.current_slice_index = selection.matched_slice_index
                    self.state.viewer.stack_view_mode = STACK_VIEW_SLICE
                if selection.is_projection_result:
                    self.state.viewer.stack_view_mode = STACK_VIEW_PROJECTION
                self.state.viewer.current_mode = VIEWER_MODE_OVERLAY
                self.input_list.setCurrentRow(matched_index)
                return

            display = self.preview_service.prepare_display_for_stem(
                viewer_state=self.state.viewer,
                input_images=self.state.session.input_images,
                stem=selection.normalized_stem,
                last_out_dir=self.state.session.last_out_dir,
                preferred_mode=VIEWER_MODE_OVERLAY,
            )
            self._apply_viewer_display(display)
            return

        self.viewer.clear("Select an image to inspect.")

    def _refresh_viewer(self) -> None:
        display = self.preview_service.prepare_display_for_input(
            viewer_state=self.state.viewer,
            input_images=self.state.session.input_images,
            current_index=self.input_list.currentRow(),
            last_out_dir=self.state.session.last_out_dir,
        )
        if display is None:
            self.viewer.clear("Select an image to inspect.")
            self._refresh_tiff_mode_label()
            return
        self._apply_viewer_display(display)

    def _apply_viewer_display(self, display: ViewerDisplayData) -> None:
        self.viewer.set_images(
            original=display.original_pixmap,
            mask=display.mask_pixmap,
            overlay=display.overlay_pixmap,
            source_mode=self.state.viewer.source_mode,
            is_stack=self.state.viewer.is_stack,
            available_slices=self.state.viewer.available_slices,
            current_slice_index=self.state.viewer.current_slice_index,
            stack_view_mode=self.state.viewer.stack_view_mode,
        )
        target_mode = display.preferred_mode or self.state.viewer.current_mode
        self.viewer.set_mode(target_mode)
        self._refresh_tiff_mode_label()

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
        if self.state.session.last_out_dir is None:
            return
        self._set_workflow_step(WORKFLOW_STEP_EXPORT)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.state.session.last_out_dir)))
