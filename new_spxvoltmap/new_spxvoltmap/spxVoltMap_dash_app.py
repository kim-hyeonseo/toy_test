# -*- coding: utf-8 -*-
# =========================================================
# base imports
# =========================================================
import numpy as np
import pandas as pd
import requests
from datetime import datetime
import threading
from threading import Timer

import geopandas as gpd
from shapely.ops import unary_union
from shapely import vectorized
from pykrige.ok import OrdinaryKriging

import dash
from dash import dcc, html
import plotly.graph_objects as go

from cryptography.fernet import Fernet
import keyring

import os, sys, configparser, flask
FERNET_KEY = b'FxitD2c1tOYnwtzOFC203_JVZzMMQvOaEaLfhH-1078='
cipher = Fernet(FERNET_KEY)


# 공용 공간 경로 설정 (예: 서버 내 다른 드라이브나 폴더)
COMMON_PATH = r"./" # 실제 GIF가 있는 경로로 수정하세요

# ===============================
# Constants
# ===============================
X_MIN, X_MAX = 125.5, 129.8
Y_MIN, Y_MAX = 34.5, 38.8

# ===============================
# Grid Resolution (Kriging)
# ===============================
GRID_NX = 140   # longitude resolution
GRID_NY = 160   # latitude resolution



# 전역 공유 저장소 (스레드 신호 및 데이터 전달용)
THREAD_STATE = {
    "is_running": False,
    "fig": None,
    "last_update": None,
    "status": "idle"  # 'idle', 'running', 'done'
}




# UI 파일: ui_file_path = os.path.join(internal_path, 'spxVoltMap.ui')
# GeoJSON: geojson_path = os.path.join(internal_path, 'land.geojson')
# CSV: csv_path = os.path.join(internal_path, 'land_coastline.csv')
# GIF: gif_path = os.path.join(internal_path, 'load_line.gif')






# =========================================================
# 환경 설정 (여기만 수정)
# =========================================================

HOST = "YOUR_HOST"
SERVICE = "YOUR_SERVICE"
USER = "YOUR_USER"
PASSWORD = "YOUR_PASSWORD"
PORT = 1521
SOURCE = "SE"   # or "MEA"




def open_browser():
    import webbrowser
    webbrowser.open("http://127.0.0.1:8050")



def load_config(base_path):
    config = configparser.ConfigParser()

    if os.path.exists(os.path.join(base_path, '../../conf/spxVoltMap_config.ini')):
        path = os.path.join(base_path, '../../conf/spxVoltMap_config.ini')
    else:
        path = os.path.join(base_path, 'spxVoltMap_config.ini')

    print(f"[CONFIG] load : {path}")
    config.read(path)

    # ---------------------------
    # select host  (--NHOST / --OHOST)
    # ---------------------------
    if len(sys.argv) > 1:
        argv = sys.argv[1]
    else:
        print("[CONFIG] host arg not found → NHOST")
        argv = '--NHOST'

    if argv == '--NHOST':
        host = config['DB']['host_naju']
        base_url = config['DB']['naju_base']
    elif argv == '--OHOST':
        host = config['DB']['host_osong']
        base_url = config['DB']['osong_base']
    else:
        raise ValueError(f"Unknown argv : {argv}")

    # ---------------------------
    # DB 계정 복호화
    # ---------------------------
    cipher = Fernet(FERNET_KEY)

    user = cipher.decrypt(config['DB']['user'].encode()).decode()
    password = cipher.decrypt(config['DB']['password'].encode()).decode()

    service = config['DB']['name']
    port = int(config['DB']['port'])
    test_mode = config['DB'].get('test_mode', 'N')

    # ---------------------------
    # voltage boundary
    # ---------------------------
    vb = config['voltage_boundary']
    voltage_cfg = {
        "graph_ov_color": vb['graph_ov_color'],
        "graph_max_color": vb['graph_max_color'],
        "graph_max_value": float(vb['graph_max_value']),
        "graph_normal_color": vb['graph_normal_color'],
        "graph_normal_value": float(vb['graph_normal_value']),
        "graph_min_color": vb['graph_min_color'],
        "graph_min_value": float(vb['graph_min_value']),
        "graph_uv_color": vb['graph_uv_color'],
        "level_density_high": int(vb['level_density_high']),
        "level_density_low": int(vb['level_density_low']),
        "level_density": int(vb['level_density']),
        "point_max_color": vb['point_max_color'],
        "point_normal_color": vb['point_normal_color'],
        "point_min_color": vb['point_min_color'],
        "point_edge_width": float(vb['point_edge_width']),
        "point_edge_color": vb['point_edge_color'],
        "point_size": float(vb['point_size']),
    }

    # ---------------------------
    # kriging parameter
    # ---------------------------
    pd_cfg = config['parameter_detail']
    kriging_cfg = {
        "model": pd_cfg['model'], 
        "slope": float(pd_cfg['slope']),
        "nugget": float(pd_cfg['nugget']),
        "scale": float(pd_cfg['scale']),
        "exponent": float(pd_cfg['exponent']),
        "range": float(pd_cfg['range']),
        "sill": float(pd_cfg['sill']),
    }

    return {
        "host": host,
        "base_url": base_url,
        "service": service,
        "user": user,
        "password": password,
        "port": port,
        "test_mode": test_mode,
        "voltage": voltage_cfg,
        "kriging": kriging_cfg,
    }

# =========================================================
# Flask-Oracle API (without Qt)
# =========================================================
FLASK_BASE_URL = ""


def set_base_url(base_url: str):
    global FLASK_BASE_URL
    FLASK_BASE_URL = base_url.rstrip("/") + "/"


def get_all_data_from_oracle(host, service, user, password, port, source="SE"):
    if source == "SE":
        query = """select * from STNVOLTMAP where STNLATITUDE != 0 and STNLONGITUDE != 0 and STNNOMINALVOLT != 0 and STNMAXVOLT != 0 and STNMINVOLT != 0 and STNAVGVOLT != 0 order by CRTTIME desc """

    else:
        query = """select * from STNVOLTMAP where STNLATITUDE != 0 and STNLONGITUDE != 0 and STNNOMINALVOLT != 0 and STNMAXVOLT_MEA != 0 and STNMINVOLT_MEA != 0 and STNAVGVOLT_MEA != 0 order by CRTTIME desc"""

    payload = {
        "params": {
                        "host": host, "port": port, "service_name": service, "user": user, "password": password, "dsn": f"{host}:{port}/{service}",
                    },
        "query": query,
    }

    try:
        resp = requests.post(
            FLASK_BASE_URL + "receive_data_elx",
            json=payload,
            timeout=10,
        )
        df = pd.DataFrame(resp.json())

        if not df.empty and "CRTTIME" in df.columns:
            df["CRTTIME"] = df["CRTTIME"].apply(
                lambda x: datetime.strptime(
                    x.replace(" GMT", ""), "%a, %d %b %Y %H:%M:%S"
                )
                if isinstance(x, str)
                else None
            )

        return df, resp.status_code

    except Exception as e:
        print("[DB ERROR]", e)
        return None, 500


def get_latest_time_from_db(host, service, user, password, port):
    # 가장 최근의 CRTTIME 1건만 가져오는 쿼리
    query = "SELECT MAX(CRTTIME) as LATEST FROM STNVOLTMAP"
    payload = {
        "params": {"host": host, "port": port, "service_name": service, "user": user, "password": password, "dsn": f"{host}:{port}/{service}"},
        "query": query,
    }
    try:
        resp = requests.post(FLASK_BASE_URL + "receive_data_elx", json=payload, timeout=5)
        data = resp.json()
        if data and len(data) > 0:
            # Flask API가 주는 시간 형식을 맞춰서 리턴 (문자열 혹은 datetime)
            return str(data[0]['LATEST'])
    except Exception as e:
        # 접속 거부(10061) 등 발생 시 그냥 None 리턴하여 에러 로그 폭발 방지
        return None

    return None



# =========================================================
# Kriging + Plotly Figure 생성
# =========================================================
def add_geojson_boundary(fig, geojson_path, line_color="lightgray"):
    gdf = gpd.read_file(geojson_path)

    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf[gdf.geometry.is_empty == False]
    gdf = gdf[gdf.is_valid]

    for geom in gdf.geometry:
        if geom.geom_type == "Polygon":
            x, y = geom.exterior.coords.xy
            fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines", line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))

        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                x, y = poly.exterior.coords.xy
                fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines", line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False,))



def build_voltmap_figure(df_mainland, df_jeju, config, value_col="STNAVGVOLT"):
    # config에서 Kriging 파라미터 추출
    k_cfg = config['kriging']
    v_cfg = config['voltage']
    
    # 모델별 파라미터 설정 (Qt 코드 로직 반영)
    variogram_params = None
    if k_cfg['model'] == 'linear':
        variogram_params = [k_cfg['slope'], k_cfg['nugget']]
    elif k_cfg['model'] in ['gaussian', 'spherical', 'exponential']:
        variogram_params = [k_cfg['range'], k_cfg['sill'], k_cfg['nugget']]
    elif k_cfg['model'] == 'power':
        variogram_params = [k_cfg['scale'], k_cfg['exponent'], k_cfg['nugget']]

    fig = go.Figure()
    # Load boundary
    add_geojson_boundary(fig, "land.geojson", line_color="#555")
    add_geojson_boundary(fig, "jeju.geojson", line_color="#555")

    # main land Kriging 
    lons = df_mainland["STNLONGITUDE"].values
    lats = df_mainland["STNLATITUDE"].values
    values = df_mainland[value_col].values


    grid_lon = np.linspace(X_MIN, X_MAX, GRID_NX)
    grid_lat = np.linspace(Y_MIN, Y_MAX, GRID_NY)


    OK = OrdinaryKriging(lons, lats, values, variogram_model=k_cfg['model'], variogram_parameters=variogram_params, verbose=False, enable_plotting=False,)

    grid_z, _ = OK.execute("grid", grid_lon, grid_lat)
    gx, gy = np.meshgrid(grid_lon, grid_lat)

    # boundary (mask용)
    land_outline = os.path.join(internal_path, 'optimized.geojson')
    gdf_land = gpd.read_file(land_outline)
    gdf_land = gdf_land[gdf_land.geometry.notnull()]
    gdf_land = gdf_land[gdf_land.geometry.is_empty == False]
    gdf_land = gdf_land[gdf_land.is_valid]
    boundary_land = unary_union(list(gdf_land.geometry))

    mask = vectorized.contains(boundary_land, gx, gy)
    grid_z[~mask] = np.nan

    ## color bar
    #fig.add_trace(go.Contour(x=grid_lon, y=grid_lat, z=grid_z, colorscale="RdBu", contours=dict(showlines=False), opacity=0.65, colorbar=dict(title="Voltage (PU)", thickness=20, len=0.75, y=1.05, orientation="h")))
    z=0

    # Jeju Kriging (without colorbar)
    lons_j = df_jeju["STNLONGITUDE"].values
    lats_j = df_jeju["STNLATITUDE"].values
    values_j = df_jeju[value_col].values

    grid_lon_j = np.linspace(X_MIN, X_MAX, GRID_NX)
    grid_lat_j = np.linspace(Y_MIN, Y_MAX, GRID_NY)

    OK_j = OrdinaryKriging(lons_j, lats_j, values_j, variogram_model=k_cfg['model'], variogram_parameters=variogram_params, verbose=False, enable_plotting=False,)
    z=0
    grid_z_j, _ = OK_j.execute("grid", grid_lon_j, grid_lat_j)
    gx_j, gy_j = np.meshgrid(grid_lon_j, grid_lat_j)

    jeju_outline = gpd.read_file(land_outline)
    gdf_jeju = gdf_jeju[gdf_jeju.geometry.notnull()]
    gdf_jeju = gdf_jeju[gdf_jeju.geometry.is_empty == False]
    gdf_jeju = gdf_jeju[gdf_jeju.is_valid]
    boundary_jeju = unary_union(list(gdf_jeju.geometry))

    mask_j = vectorized.contains(boundary_jeju, gx_j, gy_j)
    grid_z_j[~mask_j] = np.nan

    fig.add_trace(go.Contour(x=grid_lon_j, y=grid_lat_j, z=grid_z_j, colorscale="RdBu", contours=dict(showlines=False), opacity=0.65, showscale=False))


    # S/S
    fig.add_trace(go.Scatter(x=df_mainland["STNLONGITUDE"], y=df_mainland["STNLATITUDE"], mode="markers", marker=dict(size=4, color="black"), hoverinfo="skip", showlegend=False,))
    fig.add_trace(go.Scatter(x=df_jeju["STNLONGITUDE"], y=df_jeju["STNLATITUDE"], mode="markers", marker=dict(size=4, color="black"), hoverinfo="skip", showlegend=False, ))

    z=0


    fig.update_layout(
        autosize=True,

        # 본토 axis (전체)
        xaxis=dict(
            domain=[0.0, 1.0],
            range=[X_MIN, X_MAX],
            visible=False,
        ),
        yaxis=dict(
            domain=[0.0, 1.0],
            range=[Y_MIN, Y_MAX],
            visible=False,
        ),

        # 제주 inset (좌하단)
        xaxis2=dict(
            domain=[0.05, 0.28],
            range=[X_MIN, X_MAX],
            visible=False,
            anchor="y2",
        ),
        yaxis2=dict(
            domain=[0.05, 0.28],
            range=[Y_MIN, Y_MAX],
            visible=False,
            anchor="x2",
        ),

        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#1e242b",
        plot_bgcolor="#1e242b",
    )

    
    return fig


def build_colorbar_figure(vmin=1.028, vmax=1.042):
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=[[vmin, vmax]],
        colorscale="RdBu",
        showscale=True,
        colorbar=dict(
            title="Voltage (PU)",
            orientation="h",
            len=1.0,
            thickness=18,
            tickformat=".3f",
        ),
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="#1e242b",
        plot_bgcolor="#1e242b",
    )

    return fig


# =========================================================
# Dash App
# =========================================================
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

app = dash.Dash(__name__,  external_stylesheets=[dbc.themes.DARKLY], serve_locally=True)

# --- 설정 모달 레이아웃 ---
settings_modal = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("분석 및 시각화 설정"), close_button=True),
    dbc.ModalBody([
        # 1. 데이터 소스 선택 (Radio 버튼)
        html.Label("데이터 소스 선택", className="fw-bold mb-2"),
        dbc.RadioItems(
            id="source-radio",
            options=[
                {"label": "State Estimation (SE)", "value": "SE"},
                {"label": "Measurement (MEA)", "value": "MEA"},
            ],
            value="SE",
            inline=True,
            className="mb-4"
        ),

        # 3. 그래프 및 포인트 설정 (색상 및 수치)
        dbc.Row([
            # 그래프 수치 설정
            dbc.Col([
                html.Label("그래프 임계값 설정", className="fw-bold mb-2"),
                dbc.InputGroup([dbc.InputGroupText("Max"), dbc.Input(id="graph-max-val", type="number", value=1.1)], className="mb-2"),
                dbc.InputGroup([dbc.InputGroupText("Normal"), dbc.Input(id="graph-norm-val", type="number", value=1.0)], className="mb-2"),
                dbc.InputGroup([dbc.InputGroupText("Min"), dbc.Input(id="graph-min-val", type="number", value=0.9)], className="mb-2"),
                dbc.InputGroup([dbc.InputGroupText("밀도"), dbc.Input(id="level-density", type="number", value=20)], className="mb-2"),
            ], width=6),

            # 색상 선택 팔레트 (빨간색으로 색칠하셨던 부분)
            dbc.Col([
                html.Label("색상 팔레트 설정", className="fw-bold mb-2"),
                html.Div([
                    html.Label("OV/UV", className="me-2", style={"fontSize": "0.8rem"}),
                    dcc.Input(type="color", id="color-ov", value="#ff0000", className="me-2"),
                    dcc.Input(type="color", id="color-uv", value="#0000ff"),
                ], className="d-flex align-items-center mb-2"),
                html.Div([
                    html.Label("Graph (Max/Norm/Min)", className="me-2", style={"fontSize": "0.8rem"}),
                    dcc.Input(type="color", id="color-g-max", value="#ff4500", className="me-1"),
                    dcc.Input(type="color", id="color-g-norm", value="#00ff00", className="me-1"),
                    dcc.Input(type="color", id="color-g-min", value="#1e90ff"),
                ], className="mb-2"),
                html.Div([
                    html.Label("Point Size/Edge", className="me-2", style={"fontSize": "0.8rem"}),
                    dbc.Input(id="point-size", type="number", value=5, size="sm", style={"width": "60px"}, className="me-2"),
                    dcc.Input(type="color", id="color-p-edge", value="#ffffff"),
                ], className="d-flex align-items-center"),
            ], width=6),
        ]),
    ]),
    dbc.ModalFooter([
        dbc.Button("적용 (Accept)", id="btn-accept", color="primary", className="me-2"),
        dbc.Button("취소 (Cancel)", id="btn-cancel", color="secondary"),
    ]),
], id="settings-modal", is_open=False, size="lg", centered=True)



app.layout = html.Div([
    # 1. 초슬림 상단 헤더 (공간 절약형)
    dbc.Row([
        dbc.Col(html.H2("전압 분포도", className="m-0"), width="auto"),
        dbc.Col(html.Div(className="vr mx-2", style={"height": "1.5rem"}), width="auto"),
        dbc.Col(html.Span("NA", className="badge bg-secondary", style={"fontSize": "0.8rem"}), width="auto"),
        
        dbc.Col(width=True), # Spacer: 왼쪽 요소를 왼쪽으로 밀어줌



        # 체크박스 영역
        dbc.Col([
            dbc.Checklist(
                options=[{"label": "자동 갱신", "value": 1}],
                value=[1],
                id="auto-update-check",
                className="custom-check",
                inline=True,
                style={"display": "flex", "alignItems": "center"}
            )
        ], width="auto", className="me-3"),


        dbc.Col([
            dbc.Button("설정", id="open-settings", className="custom-btn me-2"),
        ], width="auto")
    ], className="header-container px-3 py-2 align-items-center"),


    # 1초마다 스레드 상태를 체크할 타이머 (Qt의 Timer 역할)
    dcc.Interval(id="timer-checker", interval=2000, n_intervals=0),

    html.Div([
        # [수정] 수행 시각과 컬러바를 한 줄에 배치
        dbc.Row([
            # 왼쪽: 수행 시각 라벨
            dbc.Col([
                html.Span("수행 시각 : ", style={"fontSize": "1.6vh", "color": "#bbb"}),
                html.Span("2026-02-03 23:00:00", id="time-display", className="time-text"),
                html.Span("(SCADA)", className="ms-2 opacity-50", style={"fontSize": "1.4vh"})
            ], width=True, className="d-flex align-items-center"),

            # 오른쪽: 컬러바(범례) 영역
            dbc.Col([
                html.Div(id="color-bar-legend", className="color-bar-container")
            ], width="auto", className="d-flex align-items-center justify-content-end")
        ], className="mb-2 align-items-center", style={"backgroundColor": "#15191c"}),

        # 메인 캔버스
        html.Div(id="main-graph-canvas", className="canvas-border", style={"height": "65vh",  "backgroundColor": "#15191c"}),
        
        # 하단 서브 캔버스
        html.Div(id="sub-graph-canvas", className="canvas-border mt-2", style={"height": "23vh",  "backgroundColor": "#15191c"})
        
    ], className="main-content m-2"),

    # 전역 로딩 오버레이 (GIF 버전)
    html.Div(
        id="global-loading-overlay",
        children=[
            html.Div([
                html.Img(
                    src="/external_assets/load_line.gif", 
                    style={"width": "250px", "height": "auto"}
                ),
            ], className="text-center")
        ],
        style={
            "position": "fixed",
            "top": 0, "left": 0, "width": "100vw", "height": "100vh",
            "backgroundColor": "rgba(0, 0, 0, 0.8)", # 배경을 조금 더 어둡게 (0.7 -> 0.8)
            "display": "none", # 초기 상태
            "alignItems": "center",
            "justifyContent": "center",
            "zIndex": 9999,
            "flexDirection": "column"
        }
    ),



    settings_modal,
    dcc.Store(id="config-store"), # 설정값을 저장할 브라우저 저장소


])


from dash import Input, Output, State
# 모달 제어 콜백

@app.callback(
    [Output("settings-modal", "is_open"),
     Output("global-loading-overlay", "style", allow_duplicate=True)], # 로딩바 추가 제어
    [Input("open-settings", "n_clicks"),
     Input("btn-accept", "n_clicks"),
     Input("btn-cancel", "n_clicks")],
    [State("settings-modal", "is_open"),
     State("config-store", "data")], # 설정값 가져오기
    prevent_initial_call='initial_duplicate'
)
def toggle_modal(n_open, n_accept, n_cancel, is_open, config):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open, no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "btn-accept":
        run_update_process(config) # 적용 누르면 바로 연산 시작
        overlay_style = {"display": "flex", "position": "fixed", "zIndex": 9999} # 로딩바 켬
        return False, overlay_style
        
    return not is_open, no_update



from dash import no_update

def run_update_process(config):
    """DB 체크 -> 로드 -> 연산 통합 프로세스"""
    global THREAD_STATE
    # 이미 실행 중이면 중복 실행 방지
    if THREAD_STATE["is_running"] or THREAD_STATE["status"] == "running": return

    # 진입하자마자 상태부터 변경해서 다음 콜백이 못 들어오게 막음
    THREAD_STATE["status"] = "running"
    THREAD_STATE["is_running"] = True

    thread = threading.Thread(
        target=data_update_and_build_worker, 
        args=(HOST, SERVICE, USER, PASSWORD, PORT, SOURCE, config)
    )
    thread.daemon = True
    thread.start()

def data_update_and_build_worker(host, service, user, pwd, port, source, config):
    global THREAD_STATE, mainland, jeju
    print("!!!! WORKER STARTED !!!!")
    try:
        THREAD_STATE["status"] = "running"
        THREAD_STATE["is_running"] = True
        new_df = None

        # [1] DB 시도
        try:
            print("[THREAD] DB 접속 시도 중...")
            new_df, status = get_all_data_from_oracle(host, service, user, pwd, port, source)
            
            # DB 응답이 정상이 아닐 경우 (None, 500 에러, 데이터 비어있음 등)
            if new_df is None or new_df.empty or status != 200:
                print(f"[THREAD] DB 데이터 부재 (Status: {status}). CSV로 전환합니다.")
                new_df = None # 확실히 비우고 다음 단계로
        except Exception as e:
            print(f"[THREAD] DB 접속 에러 발생: {e}. CSV로 전환합니다.")
            new_df = None

        # [2] DB 실패 시 CSV 로드 (강제 실행 구간)
        if new_df is None or new_df.empty:
            CSV_FILE = "voltmap1.csv"
            if os.path.exists(CSV_FILE):
                print(f"[THREAD] 로컬 CSV 로드 중: {CSV_FILE}")
                new_df = pd.read_csv(CSV_FILE)
            else:
                print("[THREAD ERROR] DB도 안되고 CSV 파일도 없습니다!")
                THREAD_STATE["status"] = "error"
                return

        # [3] 데이터 분리 및 그래프 생성
        # (여기서부터는 new_df가 DB든 CSV든 데이터를 가지고 있는 상태입니다)
        df_j = new_df[new_df["STNLATITUDE"] < 34.32].copy()
        df_m = new_df[new_df["STNLATITUDE"] >= 34.32].copy()

        fig_res = build_voltmap_figure(
            df_mainland=df_m, 
            df_jeju=df_j, 
            config=config,
            value_col="STNAVGVOLT"
        )

        # [4] 결과 저장
        THREAD_STATE["fig"] = fig_res
        THREAD_STATE["status"] = "done"
        print("[THREAD] 그래프 생성 완료.")
    
        print("!!!! WORKER FINISHED !!!!")

    except Exception as e:
        print(f"[THREAD CRITICAL ERROR] {e}")
        THREAD_STATE["status"] = "error"
    finally:
        THREAD_STATE["is_running"] = False



@app.callback(
    [Output("main-graph-canvas", "children"),
     Output("time-display", "children"),
     Output("global-loading-overlay", "style", allow_duplicate=True)],
    Input("timer-checker", "n_intervals"),
    [State("auto-update-check", "value"),
     State("time-display", "children"),
     State("config-store", "data")],
    prevent_initial_call='initial_duplicate'
)


def monitor_system(n, auto_update_val, current_display_time, config):
    global THREAD_STATE
    
    # [Case 1] 백그라운드 작업이 완료되었는지 확인 (Redraw)
    if THREAD_STATE["status"] == "done":
        THREAD_STATE["status"] = "idle"
        new_fig = THREAD_STATE["fig"]
        # 실제 데이터의 시간 혹은 현재 시간 표시
        return [dcc.Graph(figure=new_fig, style={"height": "100%"}), 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                {"display": "none"}]

    # [Case 2] 자동 갱신이 켜져 있고, 실행 중이 아닐 때 DB 체크
    if auto_update_val and not THREAD_STATE["is_running"]:
        latest_time = get_latest_time_from_db(HOST, SERVICE, USER, PASSWORD, PORT)
        
        # 시간이 달라졌다면 (DB 업데이트 발생)
        if latest_time and str(latest_time) != str(current_display_time):
            run_update_process(config) # 스레드 시작

            overlay_style = {
                    "position": "fixed", "top": 0, "left": 0, "width": "100vw", "height": "100vh",
                    "backgroundColor": "rgba(0, 0, 0, 0.8)", "display": "flex",
                    "alignItems": "center", "justifyContent": "center", "zIndex": 9999
                }
            return [no_update, no_update, overlay_style] # 로딩바 표시

    return [no_update] * 3



@app.server.route('/external_assets/<filename>')
def serve_external_assets(filename):
    # flask.send_from_directory를 사용하여 보안상 안전하게 파일을 전송합니다.
    return flask.send_from_directory(COMMON_PATH, filename)



# =========================================================
# Run Server
# =========================================================
if __name__ == "__main__":
    # 1. 서버 시작과 동시에 백그라운드 연산 스레드 딱 하나만 실행
    print("[SYSTEM] 앱 시작 - 첫 번째 데이터 로드 스레드 가동")


    if getattr(sys, 'frozen', False):
        base_path = os.getcwd().replace("\\", "/")
        sysconf_path = base_path + '/../project/EMS/conf/sysconf.ini'
        # 리소스들이 들어있는 실제 위치 (사용자 정의 기준)
        internal_path = base_path + '/HistViewer/spxVoltMap/_internal/'

        print(f"Base Path: {base_path}")
        print(f"Internal: {internal_path}")

    else:
        # 파이썬 인터프리터로 실행 시
        base_path = os.path.dirname(os.path.abspath(__file__))
        sysconf_path = os.path.dirname(os.path.abspath(__file__))
        internal_path = os.path.dirname(os.path.abspath(__file__))
        


    # 2. 설정 로드
    try:
        APP_CONFIG = load_config(internal_path)
        print("[MAIN] Config Loaded Successfully")
        
        # 설정에서 DB 정보 추출 (필요시)
        HOST = APP_CONFIG['host']
        SERVICE = APP_CONFIG['service']
        USER = APP_CONFIG['user']
        PASSWORD = APP_CONFIG['password']
        PORT = APP_CONFIG['port']
        set_base_url(APP_CONFIG['base_url']) # Flask API URL 설정
        
    except Exception as e:
        print(f"[MAIN ERROR] Config Load Failed: {e}")
        sys.exit(1)




    set_base_url("http://127.0.0.1:5000/")   # Flask 서버

    run_update_process(None) 
    
    # 2. 브라우저 자동 열기
    Timer(1, open_browser).start()
    
    # 3. 대시 서버 실행
    app.run_server(host="127.0.0.1", port=8050, debug=False) # 디버그 모드 시 스레드 2배 생성 방지




