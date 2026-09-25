from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RangeSlider
import numpy as np

from .volume import (
    Plane,
    display_to_xyz,
    extract_slice,
    finite_range,
    load_volume,
    slice_count,
)
from PeakLocalMin import PeakLocalMin


class VolumeViewer:
    ## @brief Interactive Matplotlib/Tkinter viewer for one 3D volume.
    ##
    ## The volume uses `(z, y, x)` indexing internally. Pointer coordinates
    ## are exposed consistently as `(x, y, z)` regardless of the selected plane.

    #: Color used for a selected but not yet registered pointer.
    POINTER_COLOR = "#ff3b30"
    #: Color used for the registered pointer.
    REGISTERED_COLOR = "#00c853"
    #: Color used for the local minimum result.
    LOCAL_MIN_COLOR = "#1565c0"

    def __init__(self, volume: np.ndarray | None = None, title: str = "ViewVolume") -> None:
        ## @brief Initialize the viewer state and construct the GUI.
        ## @param volume Optional numeric 3D volume in `(z, y, x)` order.
        ## @param title Window title.
        #: Source volume in ``(z, y, x)`` order.
        self.volume = None if volume is None else np.asarray(volume)
        #: Path of the volume selected by LoadData.
        self.input_path: Path | None = None
        #: Window title shown by Tkinter.
        self.title = title
        #: Currently displayed plane.
        self.plane: Plane = "axial"
        #: Zero-based index of the currently displayed slice.
        self.slice_index = 0
        #: Most recently clicked coordinate in ``(x, y, z)`` order.
        self.current_coordinate: tuple[int, int, int] | None = None
        #: Coordinate saved by the Register button.
        self.registered_coordinate: tuple[int, int, int] | None = None
        #: Local minimum coordinate found from the registered coordinate.
        self.local_minimum_coordinate: tuple[int, int, int] | None = None
        #: Whether the pointer overlay should be drawn.
        self.pointer_visible = True
        #: Guard against handling callbacks caused by programmatic slider updates.
        self._suppress_slice_callback = False

        #: Current window level used for image contrast.
        self.window_level = 0.0
        #: Current window width used for image contrast.
        self.window_width = 1.0
        #: Matplotlib range selector used to choose contrast limits.
        self.contrast_slider: RangeSlider | None = None

        #: Root Tkinter window.
        self.root = tk.Tk()
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self._finish)
        self._build_ui()
        if self.volume is not None:
            self._set_volume(self.volume, reset_pointer=False)
        else:
            self._set_data_controls_enabled(False)
            self._clear_image()

    def _build_ui(self) -> None:
        ## @brief Create widgets, the Matplotlib canvas, and event bindings.
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ttk.Frame(self.root, padding=8)
        controls.grid(row=0, column=0, sticky="ns")

        self.figure = Figure(figsize=(8, 8), dpi=100)
        layout = self.figure.add_gridspec(2, 1, height_ratios=(4, 1), hspace=0.35)
        self.axes = self.figure.add_subplot(layout[0])
        self.histogram_axes = self.figure.add_subplot(layout[1])
        self.axes.set_aspect("equal")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")

        self.load_button = ttk.Button(controls, text="LoadData", command=self._load_data)
        self.load_button.grid(row=0, column=0, pady=(0, 8), sticky="ew")

        ttk.Label(controls, text="Plane").grid(row=1, column=0, sticky="w")
        self.plane_var = tk.StringVar(value=self.plane)
        self.plane_menu = ttk.Combobox(
            controls,
            textvariable=self.plane_var,
            values=("axial", "sagittal", "coronal"),
            state="readonly",
            width=12,
        )
        self.plane_menu.grid(row=2, column=0, pady=(2, 10), sticky="ew")
        self.plane_menu.bind("<<ComboboxSelected>>", self._on_plane_changed)

        self.slice_label = ttk.Label(controls, text="Slice")
        self.slice_label.grid(row=3, column=0, sticky="w")
        self.slice_scale = tk.Scale(
            controls,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            resolution=1,
            showvalue=True,
            command=self._on_slice_changed,
            length=180,
        )
        self.slice_scale.grid(row=4, column=0, pady=(0, 10), sticky="ew")
        self.slice_scale.bind("<MouseWheel>", self._on_slice_wheel)
        self.slice_scale.bind("<Button-4>", self._on_slice_wheel)
        self.slice_scale.bind("<Button-5>", self._on_slice_wheel)

        ttk.Separator(controls).grid(row=5, column=0, sticky="ew", pady=4)
        ttk.Label(controls, text="Contrast: drag histogram limits").grid(
            row=6, column=0, sticky="w"
        )

        self.coordinate_label = ttk.Label(controls, text="Current: -", wraplength=190)
        self.coordinate_label.grid(row=7, column=0, sticky="w", pady=(8, 4))
        self.register_button = ttk.Button(controls, text="Register pointer", command=self._register)
        self.register_button.grid(row=8, column=0, pady=2, sticky="ew")
        self.peak_button = ttk.Button(
            controls, text="PeakLocalMin", command=self._find_peak_local_minimum
        )
        self.peak_button.grid(row=9, column=0, pady=2, sticky="ew")
        self.view_reset_button = ttk.Button(controls, text="ViewReset", command=self._reset_view)
        self.view_reset_button.grid(row=10, column=0, pady=2, sticky="ew")
        ttk.Button(controls, text="Finish", command=self._finish).grid(
            row=11, column=0, pady=2, sticky="ew"
        )
        self.status_label = ttk.Label(controls, text="Click an image point", wraplength=190)
        self.status_label.grid(row=12, column=0, pady=(10, 0), sticky="w")

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _set_volume(self, volume: np.ndarray, reset_pointer: bool = True) -> None:
        ## @brief Set the active volume and initialize its controls.
        ## @param volume Numeric 3D volume in `(z, y, x)` order.
        ## @param reset_pointer Whether to clear the current selection.
        self.volume = np.asarray(volume)
        minimum, maximum = finite_range(self.volume)
        self.window_level = (minimum + maximum) / 2.0
        self.window_width = max(maximum - minimum, 1.0)
        if reset_pointer:
            self.current_coordinate = None
            self.registered_coordinate = None
            self.local_minimum_coordinate = None
        self.slice_index = 0
        self._set_data_controls_enabled(True)
        self._update_plane_controls()
        self._update_histogram()
        self._update_image(reset_view=True)

    def _set_data_controls_enabled(self, enabled: bool) -> None:
        ## @brief Enable or disable controls that require loaded data.
        state = "normal" if enabled else "disabled"
        self.plane_menu.configure(state="readonly" if enabled else "disabled")
        self.slice_scale.configure(state=state)
        self.register_button.configure(state=state)
        self.peak_button.configure(state=state if self.registered_coordinate is not None else "disabled")
        self.view_reset_button.configure(state=state)

    def _load_data(self) -> None:
        ## @brief Open a file dialog and load a selected volume.
        path = filedialog.askopenfilename(
            title="Load volume data",
            filetypes=[
                ("Volume data", "*.npy *.tif *.tiff"),
                ("NumPy", "*.npy"),
                ("TIFF", "*.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            volume = load_volume(path)
        except (OSError, ValueError) as error:
            messagebox.showerror("LoadData failed", str(error), parent=self.root)
            return
        self.root.title(f"{self.title} - {Path(path).name}")
        self.input_path = Path(path)
        self.status_label.configure(text=f"Loaded: {Path(path).name}")
        self._set_volume(volume)

    def _clear_image(self) -> None:
        ## @brief Clear image and histogram axes while no data is loaded.
        self.axes.clear()
        self.histogram_axes.clear()
        self.axes.set_title("LoadData to display a volume")
        self.canvas.draw_idle()

    def _update_histogram(self) -> None:
        ## @brief Draw the volume histogram and its visual contrast selector.
        if self.volume is None:
            return
        if self.contrast_slider is not None:
            self.contrast_slider.disconnect_events()
            self.contrast_slider = None
        self.histogram_axes.clear()
        finite_values = self.volume[np.isfinite(self.volume)]
        minimum, maximum = finite_range(self.volume)
        if minimum == maximum:
            maximum = minimum + 1.0
        self.histogram_axes.hist(finite_values, bins=256, color="#607d8b")
        self.histogram_axes.set_title("Contrast / WW (drag either handle)", fontsize=9)
        self.histogram_axes.set_xlabel("voxel value", fontsize=8)
        self.histogram_axes.tick_params(labelsize=8)
        low = max(minimum, self.window_level - self.window_width / 2.0)
        high = min(maximum, self.window_level + self.window_width / 2.0)
        if low >= high:
            low, high = minimum, maximum
        self.contrast_slider = RangeSlider(
            self.histogram_axes,
            "",
            minimum,
            maximum,
            valinit=(low, high),
            valfmt="%1.3g",
            facecolor="#90caf9",
        )
        self.contrast_slider.on_changed(self._on_contrast_changed)

    def _on_contrast_changed(self, values: tuple[float, float]) -> None:
        ## @brief Apply the histogram RangeSlider limits to image contrast.
        low, high = values
        self.window_level = (low + high) / 2.0
        self.window_width = max(high - low, np.finfo(float).eps)
        if hasattr(self, "image_artist"):
            self.image_artist.set_clim(low, high)
            self.canvas.draw_idle()

    def _update_plane_controls(self) -> None:
        ## @brief Synchronize plane and slice controls with the current volume.
        ## A control is disabled when its axis has only one possible position.
        if self.volume is None:
            return
        single_frame = self.volume.shape[0] <= 1
        self.plane_menu.configure(state="disabled" if single_frame else "readonly")
        self.slice_scale.configure(state="normal")
        count = slice_count(self.volume, self.plane)
        self.slice_scale.configure(to=max(count - 1, 0))
        self.slice_scale.set(min(self.slice_index, count - 1))
        self.slice_scale.configure(state="disabled" if count <= 1 else "normal")

    def _on_plane_changed(self, _event: object = None) -> None:
        ## @brief Handle a plane selection and refresh the displayed image.
        if self.volume is None:
            return
        self.plane = self.plane_var.get()  # type: ignore[assignment]
        self.slice_index = min(self.slice_index, slice_count(self.volume, self.plane) - 1)
        self._update_plane_controls()
        self._update_image(reset_view=True)

    def _on_slice_changed(self, value: str) -> None:
        ## @brief Handle a slider change and display the selected slice.
        if self._suppress_slice_callback or self.volume is None:
            return
        self.slice_index = int(round(float(value)))
        self._update_image(reset_view=False)

    def _update_image(self, reset_view: bool) -> None:
        ## @brief Update image pixels, contrast, labels, and pointer overlays.
        ## Only the selected 2D slice is extracted, avoiding a full-volume copy
        ## whenever the user moves the slice slider.
        if self.volume is None:
            self._clear_image()
            return
        image = extract_slice(self.volume, self.plane, self.slice_index)
        low = self.window_level - self.window_width / 2.0
        high = self.window_level + self.window_width / 2.0
        if not hasattr(self, "image_artist"):
            self.image_artist = self.axes.imshow(
                image, cmap="gray", vmin=low, vmax=high, origin="upper", interpolation="nearest"
            )
        else:
            self.image_artist.set_data(image)
            self.image_artist.set_clim(low, high)
        self.axes.set_title(f"{self.plane.title()}  slice {self.slice_index}")
        self.axes.set_xlabel(self._horizontal_axis_label())
        self.axes.set_ylabel("z" if self.plane != "axial" else "y")
        if reset_view:
            self._reset_view()
        self._draw_pointer()
        self.canvas.draw_idle()

    def _reset_view(self) -> None:
        ## @brief Restore the full image extent after zooming.
        if self.volume is None:
            return
        image = extract_slice(self.volume, self.plane, self.slice_index)
        self.axes.set_xlim(-0.5, image.shape[1] - 0.5)
        self.axes.set_ylim(image.shape[0] - 0.5, -0.5)
        self.canvas.draw_idle()

    def _horizontal_axis_label(self) -> str:
        ## @brief Return the volume axis shown horizontally in the current plane.
        return {"axial": "x", "sagittal": "y", "coronal": "x"}[self.plane]

    def _draw_pointer(self) -> None:
        ## @brief Redraw the crosshair for the current logical coordinate.
        for artist in list(self.axes.lines):
            if getattr(artist, "_view_volume_pointer", False):
                artist.remove()
        if not self.pointer_visible:
            self.coordinate_label.configure(text="Current: -")
            return
        if self.current_coordinate is not None:
            x, y, z = self.current_coordinate
            color = self.REGISTERED_COLOR if self.registered_coordinate == self.current_coordinate else self.POINTER_COLOR
            self._draw_crosshair(x, y, z, color)
            value = self._voxel_value((x, y, z))
            label = f"Current: (x={x}, y={y}, z={z}), value={value:.6g}"
        else:
            label = "Current: -"
        if self.local_minimum_coordinate is not None:
            x, y, z = self.local_minimum_coordinate
            self._draw_crosshair(x, y, z, self.LOCAL_MIN_COLOR)
            value = self._voxel_value((x, y, z))
            label += f"\nPeakLocalMin: (x={x}, y={y}, z={z}), value={value:.6g}"
        self.coordinate_label.configure(text=label)

    def _voxel_value(self, coordinate: tuple[int, int, int]) -> float:
        ## @brief Return a voxel value for a logical `(x, y, z)` coordinate.
        if self.volume is None:
            raise ValueError("No volume is loaded.")
        x, y, z = coordinate
        return float(self.volume[z, y, x])

    def _draw_crosshair(self, x: int, y: int, z: int, color: str) -> None:
        ## @brief Draw one colored crosshair for a logical volume coordinate.
        display_x, display_y = self._xyz_to_display(x, y, z)
        horizontal = self.axes.axhline(display_y, color=color, linewidth=1.0)
        vertical = self.axes.axvline(display_x, color=color, linewidth=1.0)
        horizontal._view_volume_pointer = True
        vertical._view_volume_pointer = True

    def _xyz_to_display(self, x: int, y: int, z: int) -> tuple[int, int]:
        ## @brief Convert a logical `(x, y, z)` point to image coordinates.
        if self.plane == "axial":
            return x, y
        if self.plane == "sagittal":
            return y, z
        return x, z

    def _on_press(self, event: object) -> None:
        ## @brief Handle an image click and select its volume coordinate.
        if event.inaxes is not self.axes or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            if self.volume is None:
                return
            # Ctrl+drag was removed, so every normal image click displays the
            # newly selected pointer immediately.
            self.pointer_visible = True
            self.current_coordinate = display_to_xyz(
                self.plane,
                self.slice_index,
                event.xdata,
                event.ydata,
                self.volume.shape,
            )
            self.status_label.configure(text="Pointer selected. Press Register to save it.")
            self._draw_pointer()
            self.canvas.draw_idle()

    def _on_scroll(self, event: object) -> None:
        ## @brief Change slices normally, or zoom around the selected pointer with Ctrl.
        if event.inaxes is not self.axes or self.volume is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        if not self._is_control_key(event):
            self._move_slice(1 if event.step > 0 else -1)
            return
        if self.current_coordinate is None:
            self.status_label.configure(text="Select a pointer before Ctrl+wheel zoom.")
            return
        factor = 0.8 if event.step > 0 else 1.25
        # Keep the selected, unregistered pointer fixed while changing the view.
        x_min, x_max = self.axes.get_xlim()
        y_min, y_max = self.axes.get_ylim()
        x_center, y_center = self._xyz_to_display(*self.current_coordinate)
        self.axes.set_xlim(x_center + (x_min - x_center) * factor, x_center + (x_max - x_center) * factor)
        self.axes.set_ylim(y_center + (y_min - y_center) * factor, y_center + (y_max - y_center) * factor)
        self.canvas.draw_idle()

    def _on_slice_wheel(self, event: object) -> str:
        ## @brief Move the slice slider by one step for a Tkinter wheel event.
        if self.volume is None:
            return "break"
        if getattr(event, "num", None) == 4:
            direction = 1
        elif getattr(event, "num", None) == 5:
            direction = -1
        else:
            direction = 1 if event.delta > 0 else -1
        self._move_slice(direction)
        return "break"

    def _move_slice(self, direction: int) -> None:
        ## @brief Move the current slice index and synchronize the slider.
        if self.volume is None:
            return
        count = slice_count(self.volume, self.plane)
        new_index = int(np.clip(self.slice_index + direction, 0, count - 1))
        if new_index == self.slice_index:
            return
        self.slice_index = new_index
        self._suppress_slice_callback = True
        try:
            self.slice_scale.set(new_index)
        finally:
            self._suppress_slice_callback = False
        self._update_image(reset_view=False)

    @staticmethod
    def _is_control_key(event: object) -> bool:
        ## @brief Return whether a Matplotlib event contains a Ctrl modifier.
        key = str(getattr(event, "key", "") or "").lower()
        return "ctrl" in key or "control" in key

    def _register(self) -> None:
        ## @brief Save the currently selected pointer coordinate.
        if self.current_coordinate is None:
            messagebox.showwarning("No pointer", "Click a point before registering.", parent=self.root)
            return
        self.registered_coordinate = self.current_coordinate
        self.local_minimum_coordinate = None
        self.peak_button.configure(state="normal")
        self.status_label.configure(text=f"Registered: {self.registered_coordinate}")
        self._draw_pointer()
        self.canvas.draw_idle()

    def _find_peak_local_minimum(self) -> None:
        ## @brief Find and display the local minimum from the registered point.
        if self.volume is None or self.registered_coordinate is None:
            messagebox.showwarning("No registered pointer", "Register a pointer first.", parent=self.root)
            return
        x, y, z = self.registered_coordinate
        try:
            searcher = PeakLocalMin(self.volume)
            z_result, y_result, x_result = searcher.find((z, y, x))
        except (IndexError, ValueError) as error:
            messagebox.showerror("PeakLocalMin failed", str(error), parent=self.root)
            return
        self.local_minimum_coordinate = (x_result, y_result, z_result)
        self._set_slice_for_coordinate(self.local_minimum_coordinate)
        self.status_label.configure(text="PeakLocalMin coordinate found and shown in blue.")
        self._update_image(reset_view=False)

    def _set_slice_for_coordinate(self, coordinate: tuple[int, int, int]) -> None:
        ## @brief Move the current plane to the slice containing a coordinate.
        x, y, z = coordinate
        fixed_index = {"axial": z, "sagittal": x, "coronal": y}[self.plane]
        self.slice_index = fixed_index
        self._suppress_slice_callback = True
        try:
            self.slice_scale.set(fixed_index)
        finally:
            self._suppress_slice_callback = False
        self._update_plane_controls()

    def _finish(self) -> None:
        ## @brief Stop Tkinter's event loop while preserving the saved coordinate.
        self.root.quit()

    def run(self) -> tuple[int, int, int] | None:
        ## @brief Run the GUI and return the coordinate saved before closing.
        ## @return PeakLocalMin coordinate when available; otherwise the
        ## registered `(x, y, z)` coordinate, or `None`.
        self.root.mainloop()
        self.root.destroy()
        return self.local_minimum_coordinate or self.registered_coordinate


def view_volume(path: str | Path | None = None) -> tuple[tuple[int, int, int] | None, Path | None]:
    ## @brief Show the viewer and return its registered coordinate and input path.
    ## @param path Optional initial `.npy`, `.tif`, or `.tiff` path. When omitted,
    ## data is selected through the LoadData button.
    ## @return Pair of registered `(x, y, z)` coordinate and selected input path.
    volume = load_volume(path) if path is not None else None
    title = f"ViewVolume - {Path(path).name}" if path is not None else "ViewVolume"
    viewer = VolumeViewer(volume, title=title)
    viewer.input_path = Path(path) if path is not None else None
    coordinate = viewer.run()
    return coordinate, viewer.input_path
