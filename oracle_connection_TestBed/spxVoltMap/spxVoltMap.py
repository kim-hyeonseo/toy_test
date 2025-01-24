# -*- coding: cp949 -*-
from functools import partial
import sys, os, time

base_path = 'C:/Users/SYR/Desktop/fix/spxVoltMap'

from matplotlib.lines import Line2D
import pandas as pd
import psycopg2
import psycopg2.extras
import configparser

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.font_manager as fm
from matplotlib.colors import Normalize
from shapely.validation import explain_validity
# import seaborn as sns
from shapely.geometry import Point, Polygon, LineString
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import mplcursors
from PyQt5 import uic, QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QComboBox, QGraphicsView, QListView, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget, QDialog
from PyQt5.QtCore import QDate, QEvent, QObject, QTimer, pyqtSignal, Qt
# from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, griddata, Rbf, SmoothBivariateSpline, SmoothSphereBivariateSpline, CloughTocher2DInterpolator
# from pykrige.ok import OrdinaryKriging
from scipy.interpolate import Rbf
# import warnings
# warnings.filterwarnings('ignore', message='Adding colorbar to a different Figure')

from flask_server import fetch_data




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        minV, maxV = 883, 998

        ui_file_path = (os.path.join(base_path, 'spxVoltMap.ui'))
        uic.loadUi(ui_file_path, self)
        self.data_source = None  # 초기값: None

        ########### Flask의 fetch_data 함수 사용 ###########
        
        
        try:
            self.df = fetch_data()  # Flask를 통해 데이터 가져오기
            self.data_source = "Flask"  # 데이터 출처를 Flask로 설정
        except Exception as e:
            print(f"Error fetching data from Flask: {e}")
            self.df = pd.DataFrame({'ss': [], 'latitude': [], 'longitude': [], 'value': []})  # 빈 데이터로 초기화
            self.data_source = "Fallback"  # 데이터를 가져오지 못한 경우 Fallback


        print(f"Data Source: {self.data_source}")

        self.jeju = pd.DataFrame(columns=self.df.columns)

        # 콤보박스 및 버튼 설정
        self.NominalVoltage = self.findChild(QComboBox, 'NominalVoltage')
        self.RepresentativeVoltage = self.findChild(QComboBox, 'RepresentativeVoltage')
        redraw = self.findChild(QPushButton, 'redraw')
        redraw.clicked.connect(self.OptionUpdate)

        # 기본값 설정
        self.RepFlag = self.RepresentativeVoltage.currentText()

        for i in range(1, 3):
            canvas_name = f'graphCanvas{i}'
            self.prepareFigureCanvas(self.findChild(QGraphicsView, f'graphCanvas{i}'), canvas_name)
            self.drawVmap(canvas_name)

        self.show()

    def prepareFigureCanvas(self, widget, canvas_name):
        figure = plt.figure(facecolor=(32/255, 36/255, 44/255))
        canvas = FigureCanvas(figure)
        setattr(self, canvas_name, canvas)

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(canvas)

    def drawVmap(self, canvas_name):
        start_time = time.time()  # 시작 시간 측정
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        canvas = getattr(self, canvas_name)
        ax = canvas.figure.axes[0] if canvas.figure.axes else canvas.figure.add_subplot(111)

        ax.clear()
        ax.set_facecolor((32/255, 36/255, 44/255))

        # 지도 및 데이터 설정
        geojson_path = (os.path.join(base_path, 'shp.geojson'))
        tmp = gpd.read_file(geojson_path)

        if canvas_name == 'graphCanvas1':
            y_min, y_max = 33.8, 38.7
            x_min, x_max = 126, 129.8
            df = self.df
        elif canvas_name == 'graphCanvas2':
            y_min, y_max = 33.2, 33.6
            x_min, x_max = 126, 127
            df = self.jeju

        # 데이터 가져오기
        ss, lats, lons, values = df['ss'], df['latitude'], df['longitude'], df['value']

        # 색상 및 크기 설정
        colors = []
        sizes = []
        state_stable = (41/255, 255/255, 114/255)
        state_uv = (255/255, 48/255, 35/255)
        state_ov = (255/255, 130/255, 45/255)
        size = 5

        for value in values:
            if 0.98 < value <= 1.05:
                colors.append(state_uv)
                sizes.append(size * 2)
            elif value > 1.05:
                colors.append(state_ov)
                sizes.append(size * 1.5)
            else:
                colors.append(state_stable)
                sizes.append(size)

        # 그래프 데이터 시각화
        scatter = ax.scatter(lons, lats, c=colors, s=sizes, label='Data Points', zorder=5, picker=True)
        data = pd.DataFrame({'ss': ss, 'lons': lons, 'lats': lats, 'values': values})

        # 범위 설정
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # 커서 이벤트 연결
        cursor = mplcursors.cursor(scatter, hover=True)
        cursor.connect('add', lambda sel: self.on_hover_value(sel, data))

        print('Graph updated')
        canvas.draw()

        end_time = time.time()  # 종료 시간 측정
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.2f} seconds")
        QtWidgets.QApplication.restoreOverrideCursor()

    def OptionUpdate(self):
        self.RepFlag = self.RepresentativeVoltage.currentText()
        self.drawVmap(f'graphCanvas{1}')
        self.drawVmap(f'graphCanvas{2}')

    def on_hover_value(self, sel, df):
        x, y = sel.target
        row = df[(df['lons'] == x) & (df['lats'] == y)]
        if not row.empty:
            name = row['ss'].iloc[0]
            value = row['values'].iloc[0]
            sel.annotation.set_text(f'{name}: {value:.2f}')
            sel.annotation.get_bbox_patch().set_facecolor('white')
            sel.annotation.set_fontsize(10)

            sel.annotation.set_fontweight('bold')
            sel.annotation.set_fontname('Malgun Gothic')


app = QApplication(sys.argv)
Main_Window = MainWindow()
sys.exit(app.exec_())
        
 