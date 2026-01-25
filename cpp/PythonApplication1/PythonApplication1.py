import sys
import os
import numpy as np

from PyQt5 import uic
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QWidget, QVBoxLayout
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        ui_path = os.path.join(os.path.dirname(__file__), "test.ui")
        uic.loadUi(ui_path, self)

        self.btn1 = self.findChild(QPushButton, "btn1")
        self.lbl1 = self.findChild(QLabel, "label1")
        self.lbl2 = self.findChild(QLabel, "label2")

        if self.btn1 is None or self.lbl1 is None or self.lbl2 is None:
            raise RuntimeError("required widgets not found")

        self.btn1.clicked.connect(self.on_click_btn1)

        # Store baseline window size for scale factor
        self._base_w = max(1, self.width())
        self._base_h = max(1, self.height())
        self._scale = 1.0

        # Choose targets to scale
        self._font_targets = [self.btn1, self.lbl1, self.lbl2]

        # Store baseline font sizes
        self._base_font_pt = {}
        for w in self._font_targets:
            pt = w.font().pointSizeF()
            if pt <= 0:
                pt = 10.0
            self._base_font_pt[w] = pt

        # Make label2 "clickable" via eventFilter
        self.lbl2.setCursor(Qt.PointingHandCursor)
        self.lbl2.installEventFilter(self)

        # Matplotlib embed into widget1
        plot_host: QWidget = self.findChild(QWidget, "widget1")
        if plot_host is None:
            raise RuntimeError("widget1 not found")

        layout = plot_host.layout()
        if layout is None:
            layout = QVBoxLayout(plot_host)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

        self.canvas = MplCanvas(parent=plot_host, width=5, height=3, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        x = np.linspace(0, 2*np.pi, 300)
        y = np.sin(x)
        self.canvas.ax.plot(x, y)
        self.canvas.ax.grid(True)
        self.canvas.draw()

    def resizeEvent(self, event):
        # Only compute scale factor here (cheap)
        w = max(1, event.size().width())
        h = max(1, event.size().height())
        sx = w / self._base_w
        sy = h / self._base_h
        s = min(sx, sy)

        # Clamp scale to avoid extreme values
        self._scale = max(0.7, min(1.8, s))

        super().resizeEvent(event)

    def eventFilter(self, obj, event):
        # label2 click -> apply scaling
        if obj is self.lbl2 and event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                self.apply_scale(self._scale)
                return True
        return super().eventFilter(obj, event)

    def apply_scale(self, s):
        # Apply font scaling (and optionally layout spacing/margins)
        for w, base_pt in self._base_font_pt.items():
            new_pt = base_pt * s
            f = w.font()
            f.setPointSizeF(new_pt)
            w.setFont(f)

        # Optional: also scale label padding via stylesheet
        pad = int(6 * s)
        self.lbl1.setStyleSheet(f"padding:{pad}px;")
        self.lbl2.setStyleSheet(f"padding:{pad}px;")

        # Optional: update status text
        self.lbl1.setText(f"scale applied: {s:.2f}")

        # Trigger relayout
        self.centralWidget().updateGeometry()
        self.updateGeometry()

    def on_click_btn1(self):
        # Example plot update
        x = np.linspace(0, 2*np.pi, 300)
        y = np.sin(x + np.random.rand() * 2*np.pi)
        self.canvas.ax.clear()
        self.canvas.ax.plot(x, y)
        self.canvas.ax.grid(True)
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


# numpy==1.23.5
# scipy==1.9.3
# 
# pandas==1.5.3
# matplotlib==3.7.2
# mplcursors==0.5.2
# 
# psycopg2-binary==2.9.9
# 
# shapely==1.8.5
# geopandas==0.12.2
# 
# PyQt5==5.15.9
# 
# pyinstaller==5.13.2
