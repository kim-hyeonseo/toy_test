# -*- coding: utf-8 -*-
from functools import partial
import sys
import os
import time

if getattr(sys, "frozen", False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QWidget, QLayout
from PyQt5.QtCore import QSize
import configparser
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Normalize
from shapely.validation import explain_validity
from shapely.geometry import Point, Polygon, LineString
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.interpolate import Rbf
import mplcursors

from PyQt5.QtGui import QFont
from PyQt5.QtCore import QTimer, QEvent
from PyQt5.QtWidgets import QLabel, QPushButton, QComboBox, QTextEdit, QLineEdit, QCheckBox, QRadioButton, QGroupBox

from PyQt5 import uic, QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QListView, QMainWindow,
    QPushButton, QTextEdit, QVBoxLayout, QWidget, QDialog
)
from PyQt5.QtCore import QDate, QEvent, QObject, QTimer, pyqtSignal, Qt

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None


class NumericEventFilter(QObject):
    def eventFilter(self, source, event):
        if event.type() != QEvent.KeyPress:
            return super().eventFilter(source, event)

        key = event.key()
        mods = event.modifiers()

        if mods & Qt.ControlModifier:
            if key in (Qt.Key_C, Qt.Key_V, Qt.Key_X, Qt.Key_A, Qt.Key_Z, Qt.Key_Y):
                return False

        allowed = (
            (Qt.Key_0 <= key <= Qt.Key_9) or
            key in (Qt.Key_Period, Qt.Key_Minus) or
            key in (
                Qt.Key_Backspace, Qt.Key_Delete, Qt.Key_Left, Qt.Key_Right,
                Qt.Key_Home, Qt.Key_End, Qt.Key_Tab
            )
        )

        if not allowed:
            return True

        if hasattr(source, "toPlainText"):
            text = source.toPlainText()
        elif hasattr(source, "text"):
            text = source.text()
        else:
            text = ""

        if key == Qt.Key_Period and "." in text:
            return True

        if key == Qt.Key_Minus:
            if "-" in text:
                return True
            if hasattr(source, "textCursor"):
                if source.textCursor().position() != 0:
                    return True
            else:
                return True

        return False





def make_dummy_voltage_df(n=260, seed=1234):
    rng = np.random.default_rng(seed)

    uids = np.arange(1, n + 1, dtype=int)
    labels = [f"SS{uid:04d}" for uid in uids]

    is_jeju = rng.random(n) < 0.22

    lat = np.where(is_jeju, rng.uniform(33.2, 33.6, n), rng.uniform(34.0, 38.6, n))
    lon = np.where(is_jeju, rng.uniform(126.0, 127.0, n), rng.uniform(126.0, 129.8, n))

    nominal = rng.choice([765, 345, 154], size=n, p=[0.08, 0.35, 0.57])

    avg = rng.normal(1.0, 0.02, n)
    avg = np.clip(avg, 0.90, 1.12)

    spread = rng.uniform(0.01, 0.03, n)
    minv = np.clip(avg - spread, 0.0, 1.2)
    maxv = np.clip(avg + spread, 0.0, 1.2)

    zero_mask = rng.random(n) < 0.03
    avg[zero_mask] = 0.0
    minv[zero_mask] = 0.0
    maxv[zero_mask] = 0.0

    now = pd.Timestamp.now().floor("s")
    crttime = [now + pd.Timedelta(seconds=int(i)) for i in range(n)]

    df = pd.DataFrame(
        {
            "uid": uids,
            "stnlabel": labels,
            "stnlatitude": lat,
            "stnlongitude": lon,
            "stnnominalvolt": nominal,
            "stnmaxvolt": maxv,
            "stnminvolt": minv,
            "stnavgvolt": avg,
            "crttime": crttime,
        }
    )
    return df


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        ui_file_path = os.path.join(base_path, "spxVoltMap.ui")
        uic.loadUi(ui_file_path, self)



        # scale_
        # inside MainWindow.__init__ after loadUi
        self._snapshot_done = False
        self._base_w = 1
        self._base_h = 1
        self._scale = 1.0
        self._tooltip_base_fs = 12.0
        self._last_hover_sel = None
        self._base_font_pt = {}
        self._font_targets = []


        self.resize_trigger = self.findChild(QLabel, "resize_trigger")
        if self.resize_trigger is not None:
            self.resize_trigger.setCursor(Qt.PointingHandCursor)
            self.resize_trigger.installEventFilter(self)

        # call after show
        QTimer.singleShot(0, self.snapshot_fonts)






        self.high_lower = 1.05
        self.low_upper = 0.98
        self.low_lower = 0.95

        self.df_all = make_dummy_voltage_df(n=260, seed=1234)

        is_jeju = self.df_all["stnlatitude"] < 34.32
        self.df = self.df_all[~is_jeju].copy()
        self.jeju = self.df_all[is_jeju].copy()

        self.NominalVoltage = self.findChild(QComboBox, "NominalVoltage")
        self.RepresentativeVoltage = self.findChild(QComboBox, "RepresentativeVoltage")
        self.Unit = self.findChild(QComboBox, "Unit")

        redraw = self.findChild(QPushButton, "redraw")
        setting_btn = self.findChild(QPushButton, "setting_btn")

        if redraw is not None:
            redraw.clicked.connect(self.OptionUpdate)
        if setting_btn is not None:
            setting_btn.clicked.connect(self.Open_setting)

        if self.Unit is not None:
            self.Unit.setEnabled(False)

        if self.NominalVoltage is not None:
            self.NominalVoltage.currentTextChanged.connect(self.UnitEnable)

        combo_box_group = [x for x in [self.NominalVoltage, self.RepresentativeVoltage, self.Unit] if x is not None]
        combo_box_style = """
            QComboBox{
                background-color : rgb(28,33,38);
                color : white;
                font-size : 14px;
                font-family : 'Malgun Gothic';
                font-weight : bold;
                border : 1px solid gray;
                border-radius: 3px;
                padding: 5px;
            }

            QComboBox QListView{
                background-color : rgb(0,167,117);
                selection-background-color : rgb(0,123,192);
                color : white;
                font-size : 12px;
                font-family : 'Malgun Gothic';
                font-weight : bold;
            }
        """

        for obj in combo_box_group:
            obj.setStyleSheet(combo_box_style)
            lv = QListView()
            obj.setView(lv)
        
        # scale_ 
        self._combo_base_qss = combo_box_style


        self.NominalFlag = self.NominalVoltage.currentText() if self.NominalVoltage is not None else "ALL"
        self.RepFlag = self.RepresentativeVoltage.currentText() if self.RepresentativeVoltage is not None else "avr"
        self.UnitFlag = self.Unit.currentText() if self.Unit is not None else "PU"

        self.drag_start_pos = {}

        self._cbar_axes = {}

        for i in range(1, 3):
            canvas_name = f"graphCanvas{i}"
            w = self.findChild(QWidget, canvas_name)
            if w is None:
                continue
            self.prepareFigureCanvas(w, canvas_name)
            self.drawVmap(canvas_name)

        self.colorBar = self.findChild(QWidget, "colorBar")

        self.show()

    def Open_setting(self):
        setting = QDialog(self)
        setting.setWindowFlags(Qt.WindowCloseButtonHint)

        ui_file_path = os.path.join(base_path, "setting.ui")
        uic.loadUi(ui_file_path, setting)


        # scale_
        base_font_pt, base_layout, base_size = self.snapshot_dialog_layout(setting)
        self.apply_dialog_scale_layout(
            setting,
            getattr(self, "_scale", 1.0),
            base_font_pt,
            base_layout,
            base_size,
        )

        # scale_  __ end

        high_lower = setting.findChild(QTextEdit, "high_lower")
        normal_upper = setting.findChild(QTextEdit, "normal_upper")
        normal_lower = setting.findChild(QTextEdit, "normal_lower")
        low_upper = setting.findChild(QTextEdit, "low_upper")
        low_lower = setting.findChild(QTextEdit, "low_lower")

        inputlimit = NumericEventFilter()
        for w in [high_lower, normal_upper, normal_lower, low_upper, low_lower]:
            if w is not None:
                w.installEventFilter(inputlimit)

        if high_lower is not None:
            high_lower.setText(f"{self.high_lower}")
        if normal_upper is not None:
            normal_upper.setText(f"{self.high_lower}")
        if normal_lower is not None:
            normal_lower.setText(f"{self.low_upper}")
        if low_upper is not None:
            low_upper.setText(f"{self.low_upper}")
        if low_lower is not None:
            low_lower.setText(f"{self.low_lower}")

        if high_lower is not None and normal_upper is not None:
            high_lower.textChanged.connect(partial(self.syncText, high_lower, normal_upper))
            normal_upper.textChanged.connect(partial(self.syncText, normal_upper, high_lower))

        if normal_lower is not None and low_upper is not None:
            normal_lower.textChanged.connect(partial(self.syncText, normal_lower, low_upper))
            low_upper.textChanged.connect(partial(self.syncText, low_upper, normal_lower))

        setting.exec_()


    def syncText(self, source, target):
        text = source.toPlainText()
        if text != target.toPlainText():
            target.blockSignals(True)
            target.setPlainText(text)
            target.blockSignals(False)

    def prepareFigureCanvas(self, widget, canvas_name):
        figure = plt.figure(facecolor=(32 / 255, 36 / 255, 44 / 255))
        canvas = FigureCanvas(figure)
        setattr(self, canvas_name, canvas)


        # scale_
        layout = widget.layout()
        if layout is None:
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
        layout.addWidget(canvas)

        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(canvas)

    def drawVmap(self, canvas_name):
        start_time = time.time()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        canvas = getattr(self, canvas_name)
        if canvas.figure.axes:
            ax = canvas.figure.axes[0]
            while len(canvas.figure.axes) > 1:
                canvas.figure.axes[-1].remove()
        else:
            ax = canvas.figure.add_subplot(111)

        ax.clear()
        ax.set_facecolor((32 / 255, 36 / 255, 44 / 255))

        if canvas_name == "graphCanvas1":
            ax.set_box_aspect(1.23)
            canvas.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        elif canvas_name == "graphCanvas2":
            ax.set_box_aspect(0.5)
            canvas.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)

        if canvas_name == "graphCanvas1":
            y_min, y_max = 33.8, 38.7
            x_min, x_max = 126.0, 129.8
            df = self.df
            csv_name = "land_coastline.csv"
        else:
            y_min, y_max = 33.2, 33.6
            x_min, x_max = 126.0, 127.0
            df = self.jeju
            csv_name = "jeju_coastline.csv"

        tmp = None
        df_places = None
        try:
            geojson_path = os.path.join(base_path, "shp.geojson")
            tmp = gpd.read_file(geojson_path)
            keywords = ["Jeju-si", "Seogwipo-si"]
            jeju_mask = tmp["SIG_ENG_NM"].str.contains("|".join(keywords))
            if canvas_name == "graphCanvas1":
                df_places = tmp[~jeju_mask]
            else:
                df_places = tmp[jeju_mask]
        except Exception:
            df_places = None

        if df_places is not None:
            df_places.plot(ax=ax, color="white", edgecolor="lightgray", markersize=4)

        rep = "stnavgvolt"
        if self.RepFlag == "max":
            rep = "stnmaxvolt"
        elif self.RepFlag == "min":
            rep = "stnminvolt"
        elif self.RepFlag == "avr":
            rep = "stnavgvolt"

        if self.NominalFlag == "ALL":
            dff = df[df[rep] != 0].copy()
            ss = dff["stnlabel"]
            lats = dff["stnlatitude"]
            lons = dff["stnlongitude"]
            values4color = dff[rep]
            values = values4color
        else:
            if self.NominalFlag in ("765kV", "345kV", "154kV"):
                ref_kv = int(self.NominalFlag[:3])
            else:
                ref_kv = 345

            dff = df[(df["stnnominalvolt"] == ref_kv) & (df[rep] != 0)].copy()
            ss = dff["stnlabel"]
            lats = dff["stnlatitude"]
            lons = dff["stnlongitude"]
            values4color = dff[rep]

            if self.UnitFlag == "kV":
                values = values4color * ref_kv
            else:
                values = values4color

        if len(values) < 3:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            canvas.draw()
            QtWidgets.QApplication.restoreOverrideCursor()
            return

        colors = []
        sizes = []
        state_stable = (41 / 255, 255 / 255, 114 / 255)
        state_uv = (255 / 255, 48 / 255, 35 / 255)
        state_ov = (255 / 255, 130 / 255, 45 / 255)
        state_zero = (0.0, 0.0, 0.0)
        base_size = 5

        for v in values4color:
            if v == 0:
                colors.append(state_zero)
                sizes.append(base_size)
            elif self.low_lower < v <= self.low_upper:
                colors.append(state_uv)
                sizes.append(base_size * 2)
            elif v > self.high_lower:
                colors.append(state_ov)
                sizes.append(base_size * 1.5)
            else:
                colors.append(state_stable)
                sizes.append(base_size)

        scatter = ax.scatter(lons, lats, c=colors, s=sizes, zorder=5, picker=True)

        data = pd.DataFrame(
            {
                "ss": ss.values,
                "lons": lons.values,
                "lats": lats.values,
                "values": values.values if hasattr(values, "values") else np.array(values),
            }
        )

        grid_lon = np.linspace(x_min, x_max, 80)
        grid_lat = np.linspace(y_min, y_max, 80)
        grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

        inner_points = None
        try:
            csv_path = os.path.join(base_path, csv_name)
            outline_land = pd.read_csv(csv_path)
            if "lon_g" in outline_land.columns and "lat_g" in outline_land.columns:
                inner_points = list(zip(outline_land["lon_g"], outline_land["lat_g"]))
        except Exception:
            inner_points = None

        if inner_points is None or len(inner_points) < 3:
            inner_points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

        try:
            inner_boundary = Polygon(inner_points)
            if not inner_boundary.is_valid:
                inner_boundary = inner_boundary.buffer(0)
        except Exception:
            inner_boundary = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])

        try:
            rbf = Rbf(lons, lats, values, function="multiquadric", smooth=0.05)
            grid_z = rbf(grid_x, grid_y)
        except Exception:
            grid_z = np.full_like(grid_x, np.nan, dtype=float)

        try:
            from matplotlib.path import Path
            path = Path(np.array(inner_points))
            pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            mask = path.contains_points(pts).reshape(grid_x.shape)
            grid_z = np.where(mask, grid_z, np.nan)
        except Exception:
            pass

        min_margin, max_margin = 0.95, 1.05
        vmin = float(np.nanmin(values)) * min_margin
        vmax = float(np.nanmax(values)) * max_margin
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.95, 1.10

        contour_levels = np.linspace(vmin, vmax, 100)
        contour = ax.contourf(
            grid_x,
            grid_y,
            grid_z,
            levels=contour_levels,
            cmap="coolwarm",
            alpha=0.5,
            zorder=1,
            norm=Normalize(vmin=0.95, vmax=1.1),
        )


        # scale_
        mpl_fs = 12.0 * getattr(self, "_scale", 1.0)


        if canvas_name == "graphCanvas1":
            ticks = np.array([0.95, 1.0, 1.05, 1.1])
            cbar_ax = canvas.figure.add_axes([0.9, 0.18, 0.02, 0.68])
            cbar = ax.figure.colorbar(contour, ax=ax, cax=cbar_ax, ticks=ticks, orientation="vertical")
            cbar.ax.set_yticklabels(
                ticks,
                fontdict={"fontsize": mpl_fs, "fontweight": "bold", "fontname": "Malgun Gothic", "color": "white"},   # scale_
            )

            legend_edge = (32 / 255, 36 / 255, 44 / 255)
            legend_elements = [
                Line2D([0], [0], marker="o", color=legend_edge, markerfacecolor=state_ov, label="OV", markersize=8),
                Line2D([0], [0], marker="o", color=legend_edge, markerfacecolor=state_uv, label="UV", markersize=8),
                Line2D([0], [0], marker="o", color=legend_edge, markerfacecolor=state_zero, label="ZERO", markersize=8),
                Line2D([0], [0], marker="o", color=legend_edge, markerfacecolor=state_stable, label="OK", markersize=8),
            ]
            legend = ax.legend(
                handles=legend_elements,
                loc="best",
                frameon=True,
                facecolor=(32 / 255, 36 / 255, 44 / 255),
                edgecolor="lightgray",
                prop={"size": mpl_fs, "weight": "bold", "family": "Malgun Gothic"}, #scale_
            )
            for t in legend.get_texts():
                t.set_color("white")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        cursor = mplcursors.cursor(scatter, hover=True)
        cursor.connect("add", lambda sel: self.on_hover_value(sel, data, self.UnitFlag))

        canvas.draw()

        elapsed = time.time() - start_time
        QtWidgets.QApplication.restoreOverrideCursor()

    def OptionUpdate(self):
        self.NominalFlag = self.NominalVoltage.currentText() if self.NominalVoltage is not None else "ALL"
        self.RepFlag = self.RepresentativeVoltage.currentText() if self.RepresentativeVoltage is not None else "avr"
        self.UnitFlag = self.Unit.currentText() if self.Unit is not None else "PU"

        self.drawVmap("graphCanvas1")
        self.drawVmap("graphCanvas2")

    def UnitEnable(self):
        if self.NominalVoltage is None or self.Unit is None:
            return
        if self.NominalVoltage.currentText() == "ALL":
            self.Unit.setEnabled(False)
        else:
            self.Unit.setEnabled(True)

    def connect_event_handlers(self):
        for i in range(1, 3):
            name = f"graphCanvas{i}"
            if not hasattr(self, name):
                continue
            graph_canvas = getattr(self, name)
            graph_canvas.mpl_connect("scroll_event", self.on_scroll)
            graph_canvas.mpl_connect("button_press_event", self.on_press)
            graph_canvas.mpl_connect("button_release_event", self.on_release)
            graph_canvas.mpl_connect("motion_notify_event", self.on_motion)

    def on_scroll(self, event):
        ax = event.inaxes
        if ax is None:
            return

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if event.step > 0:
            ax.set_xlim(xlim[0] + (xlim[1] - xlim[0]) * 0.1, xlim[1] - (xlim[1] - xlim[0]) * 0.1)
            ax.set_ylim(ylim[0] + (ylim[1] - ylim[0]) * 0.1, ylim[1] - (ylim[1] - ylim[0]) * 0.1)
        else:
            ax.set_xlim(xlim[0] - (xlim[1] - xlim[0]) * 0.1, xlim[1] + (xlim[1] - xlim[0]) * 0.1)
            ax.set_ylim(ylim[0] - (ylim[1] - ylim[0]) * 0.1, ylim[1] + (ylim[1] - ylim[0]) * 0.1)

        y_range = abs(ylim[1] - ylim[0])
        if y_range < 100:
            interval = 10
        else:
            interval = np.ceil(y_range / 5)
            interval = max(5, 5 * round(interval / 5))

        ax.yaxis.set_major_locator(MultipleLocator(interval))
        ax.figure.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes is not None:
            self.drag_start_pos[event.inaxes] = (event.xdata, event.ydata)

    def on_release(self, event):
        if event.inaxes in self.drag_start_pos:
            del self.drag_start_pos[event.inaxes]

    def on_motion(self, event):
        if event.inaxes in self.drag_start_pos and event.xdata is not None and event.ydata is not None:
            start_x, start_y = self.drag_start_pos[event.inaxes]
            dx = event.xdata - start_x
            dy = event.ydata - start_y

            ax = event.inaxes
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            ax.set_ylim(ylim[0] - dy, ylim[1] - dy)

            self.drag_start_pos[event.inaxes] = (event.xdata, event.ydata)
            event.canvas.draw_idle()

    def on_hover_value(self, sel, df, unit):
        import warnings
        warnings.filterwarnings("ignore", message="Pick support for QuadContourSet is missing.")
        x, y = sel.target

        row = df[(df["lons"] == x) & (df["lats"] == y)]
        if len(row):
            name = row["ss"].tolist()[0]
            value = row["values"].tolist()[0]
            sel.annotation.set(text=f"{name} {value} {unit}")

            sel.annotation.get_bbox_patch().set_facecolor("white")
            # scale_
            #sel.annotation.set_fontsize(12)
            fs = self._tooltip_base_fs * getattr(self, "_scale", 1.0)
            sel.annotation.set_fontsize(fs)



            self._last_hover_sel = sel


            sel.annotation.set_fontweight("bold")
            sel.annotation.set_fontname("Malgun Gothic")




    # scale_
    def snapshot_fonts(self):
        if self._snapshot_done:
            return

        sz = self.size()
        self._base_w = max(1, sz.width())
        self._base_h = max(1, sz.height())

        targets = []
        targets += self.findChildren(QLabel)
        targets += self.findChildren(QPushButton)
        targets += self.findChildren(QComboBox)
        targets += self.findChildren(QTextEdit)
        targets += self.findChildren(QLineEdit)
        targets += self.findChildren(QCheckBox)
        targets += self.findChildren(QRadioButton)
        targets += self.findChildren(QGroupBox)

        uniq = []
        seen = set()
        for w in targets:
            if w is None:
                continue
            if id(w) in seen:
                continue
            seen.add(id(w))
            uniq.append(w)

        self._font_targets = uniq
        self._base_font_pt = {}

        for w in self._font_targets:
            f = w.font()
            pt = f.pointSizeF()
            if pt <= 0:
                pt = 10.0
            self._base_font_pt[w] = pt

        self._snapshot_done = True


    def resizeEvent(self, event):
        if self._snapshot_done:
            w = max(1, event.size().width())
            h = max(1, event.size().height())
            sx = w / self._base_w
            sy = h / self._base_h
            s = min(sx, sy)

            # clamp
            if s < 0.7:
                s = 0.7
            elif s > 1.8:
                s = 1.8

            self._scale = s

        super().resizeEvent(event)


    def eventFilter(self, obj, event):
        if obj is self.resize_trigger and event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                self.apply_scale(self._scale)
                return True
        return super().eventFilter(obj, event)


    def apply_scale(self, s):
        if not self._snapshot_done:
            self.snapshot_fonts()

        for w in self._font_targets:
            base_pt = self._base_font_pt.get(w, None)
            if base_pt is None:
                continue

            new_pt = base_pt * s
            f = w.font()
            if abs(f.pointSizeF() - new_pt) < 0.25:
                continue
            f.setPointSizeF(new_pt)
            w.setFont(f)

        if getattr(self, "_last_hover_sel", None) is not None:
            sel = self._last_hover_sel
            try:
                fs = self._tooltip_base_fs * getattr(self, "_scale", 1.0)
                sel.annotation.set_fontsize(fs)
                sel.annotation.figure.canvas.draw_idle()
            except Exception:
                pass


        self.apply_combo_scale(s)



        self.centralWidget().updateGeometry()
        self.updateGeometry()
        if hasattr(self, "graphCanvas1"):
            self.drawVmap("graphCanvas1")
        if hasattr(self, "graphCanvas2"):
            self.drawVmap("graphCanvas2")






    def snapshot_dialog_layout(self, dlg):
        targets = dlg.findChildren(QWidget)
        targets.append(dlg)

        base_font_pt = {}
        for w in targets:
            pt = w.font().pointSizeF()
            if pt <= 0:
                pt = 10.0
            base_font_pt[w] = pt

        base_layout = {}

        def walk_layout(layout):
            if layout is None:
                return
            if id(layout) in base_layout:
                return

            base_layout[id(layout)] = (
                layout.contentsMargins(),
                layout.spacing()
            )

            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item is None:
                    continue
                child_layout = item.layout()
                if child_layout is not None:
                    walk_layout(child_layout)

        walk_layout(dlg.layout())

        base_size = dlg.size()
        if base_size.width() <= 0 or base_size.height() <= 0:
            base_size = QSize(400, 300)

        return base_font_pt, base_layout, base_size


    def apply_dialog_scale_layout(self, dlg, s, base_font_pt, base_layout, base_size):
        s = max(0.7, min(1.8, float(s)))

        dlg.resize(int(base_size.width() * s), int(base_size.height() * s))

        for w, pt in base_font_pt.items():
            f = w.font()
            f.setPointSizeF(pt * s)
            w.setFont(f)

        def walk_layout(layout):
            if layout is None:
                return
            key = id(layout)
            if key in base_layout:
                margins, spacing = base_layout[key]
                layout.setContentsMargins(
                    int(margins.left() * s),
                    int(margins.top() * s),
                    int(margins.right() * s),
                    int(margins.bottom() * s),
                )
                if spacing >= 0:
                    layout.setSpacing(int(spacing * s))

            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item is None:
                    continue
                child_layout = item.layout()
                if child_layout is not None:
                    walk_layout(child_layout)

        walk_layout(dlg.layout())

        dlg.adjustSize()


    def _scaled_qss_px(self, qss, s):
        import re
        def repl(m):
            px = int(m.group(1))
            return f"font-size : {max(8, int(px * s))}px"
        return re.sub(r"font-size\s*:\s*(\d+)\s*px", repl, qss)

    def apply_combo_scale(self, s):
        if not hasattr(self, "_combo_base_qss"):
            return
        qss = self._scaled_qss_px(self._combo_base_qss, s)

        combos = self.findChildren(QComboBox)
        for cb in combos:
            cb.setStyleSheet(qss)

            v = cb.view()
            if v is not None:
                v.setStyleSheet(qss)

            base_pt = self._base_font_pt.get(cb, cb.font().pointSizeF())
            if base_pt <= 0:
                base_pt = 10.0

            f = cb.font()
            f.setPointSizeF(base_pt * s)
            cb.setFont(f)

            if v is not None:
                vf = v.font()
                vf.setPointSizeF(base_pt * s)
                v.setFont(vf)





if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec_())
