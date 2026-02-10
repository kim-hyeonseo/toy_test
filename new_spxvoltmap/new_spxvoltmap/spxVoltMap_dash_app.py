# -*- coding: utf-8 -*-
# base imports
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

import os
from flask import request
from dash import Dash, html, Input, Output, clientside_callback

# def generate_new_key():
#     key = Fernet.generate_key()
#     print(f"[신규 키 생성 완료]: {key.decode()}")
#     print(" 이 키를 복사해서 프로그램의 'FERNET_KEY' 변수에 붙여넣으세요.\n")
#     return key
# 
# def encrypt_text(plain_text, key):
#     """문자열을 암호화합니다."""
#     cipher = Fernet(key)
#     encrypted_text = cipher.encrypt(plain_text.encode())
#     return encrypted_text.decode()
# 
# def decrypt_text(encrypted_text, key):
#     """암호화된 문자열을 복호화합니다."""
#     cipher = Fernet(key)
#     decrypted_text = cipher.decrypt(encrypted_text.encode())
#     return decrypted_text.decode()
# 
# 
# key = generate_new_key()
# enc_id = encrypt_text('SE_USER', key)
# enc_pw = encrypt_text('RJfoth12#', key)

FERNET_KEY = b'3vqHCJxl_vDg-QA2L2ZilVUnObnsw__JvCE7ZWIKK3E='
cipher = Fernet(FERNET_KEY)






class GlobalVars:
    # 1. 콤보박스 및 주요 플래그 기본값
    nominal_flag = "ALL"  # NominalVoltage 기본값
    rep_flag = "최대"      # RepresentativeVoltage 기본값
    unit_flag = "PU"       # Unit 기본값
    current_source = "SE"
        
    # 2. DB 및 연결 정보
    host = None
    base_url = None
    name = None
    user = None
    password = None
    port = None
    test_mode = None
    source = 'SE' # or MEA
        
    # 3. 리소스 및 기타 설정
    name_ref = None
    voltage_cfg = {}
    kriging_cfg = {}
        
    # 4. 경로 설정 (전역 유지)
    base_path = os.getcwd().replace("\\", "/")
    base_path = os.path.join(base_path, 'HistViewer/spxVoltMap/_internal/').replace("\\", "/")
    sysconf_path = os.path.join(base_path, '../project/EMS/conf/sysconf.ini').replace("\\", "/")


    # Constants
    LAND_XMIN, LAND_XMAX = 125.5, 129.8
    LAND_YMIN, LAND_YMAX = 34.2, 38.8


    JEJU_XMIN, JEJU_XMAX = 126, 127
    JEJU_YMIN, JEJU_YMAX = 33.2, 33.6


    # Grid Resolution (Kriging)
    GRID_RESOL = 100   

    df_raw = None


# 전역 인스턴스 생성
Vars = GlobalVars()


# 전역 공유 저장소 (스레드 신호 및 데이터 전달용)
THREAD_STATE = {
    "is_running": False,
    "fig": None,
    "last_update": None,
    "status": "idle"  # 'idle', 'running', 'done'
}

# exe 가 바로 웹 열게 하는 부분
def open_browser():
    import webbrowser
    webbrowser.open("http://127.0.0.1:8050")

def load_config():
    config = configparser.ConfigParser()
    if os.path.exists(os.path.join(Vars.base_path, '../../conf/spxVoltMap_config.ini')):
        path = os.path.join(Vars.base_path, '../../conf/spxVoltMap_config.ini')
    else:
        path = os.path.join(Vars.base_path, 'spxVoltMap_config.ini')

    print(f"[CONFIG] load : {path}")
    config.read(path)

    if len(sys.argv) > 1:
        argv = sys.argv[1]
    else:
        argv = '--OHOST'
        print(f"[CONFIG] host arg not found → {argv[2:]}")

    if argv == '--NHOST':
        Vars.host = config['DB']['host_naju']
        base_url = config['DB']['naju_base']
    elif argv == '--OHOST':
        Vars.host = config['DB']['host_osong']
        base_url = config['DB']['osong_base']
    else:
        raise ValueError(f"Unknown argv : {argv}")

    # ---------------------------
    # DB 계정 복호화
    # ---------------------------
    cipher = Fernet(FERNET_KEY)
    Vars.base_url=base_url.strip("'")

    Vars.user = cipher.decrypt(config['DB']['user'].encode()).decode()
    Vars.password = cipher.decrypt(config['DB']['password'].encode()).decode()

    Vars.name = config['DB']['name']
    Vars.port = config['DB']['port']
    Vars.test_mode = config['DB'].get('test_mode', 'N')

    # voltage boundary
    vb = config['voltage_boundary']
    Vars.voltage_cfg = {
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

    # kriging parameter
    pd_cfg = config['parameter_detail']
    Vars.kriging_cfg = {
        "model": pd_cfg['model'], 
        "slope": float(pd_cfg['slope']),
        "nugget": float(pd_cfg['nugget']),
        "scale": float(pd_cfg['scale']),
        "exponent": float(pd_cfg['exponent']),
        "range": float(pd_cfg['range']),
        "sill": float(pd_cfg['sill']),
    }
    print("[CONFIG] App config loaded successfully.")

# =========================================================
# Flask-Oracle API (without Qt)
# =========================================================

def get_all_data_from_oracle(source="SE"):
    if source == "SE":
        query = """select * from STNVOLTMAP where STNLATITUDE != 0 and STNLONGITUDE != 0 and STNNOMINALVOLT != 0 and STNMAXVOLT != 0 and STNMINVOLT != 0 and STNAVGVOLT != 0 order by CRTTIME desc """

    else:
        query = """select * from STNVOLTMAP where STNLATITUDE != 0 and STNLONGITUDE != 0 and STNNOMINALVOLT != 0 and STNMAXVOLT_MEA != 0 and STNMINVOLT_MEA != 0 and STNAVGVOLT_MEA != 0 order by CRTTIME desc"""

    payload = {
        "params": {"host": Vars.host, "port": Vars.port, "service_name": Vars.name, "user": Vars.user, "password": Vars.password, "dsn": f"{Vars.host}:{Vars.port}/{Vars.name}"},
        "query": query,
    }


    try:    
        plus_url = 'receive_data_elx'
        dsn = f'{Vars.base_url}'

        resp = requests.post(dsn + plus_url, json=payload)
        print('flag2')
        df = pd.DataFrame(resp.json())
        print('flag3')
        if len(df)>0 : df["CRTTIME"] = df['CRTTIME'].apply(lambda x: datetime.strptime(x.replace(' GMT', ''), "%a, %d %b %Y %H:%M:%S") if isinstance(x, str) else None)
        else : df=[]
        print('flag4')
    except :
        df=[] 
        print('flag5')
    if df==[]:
        # DB 실패 또는 데이터가 없을 경우 CSV 로드 (Fallback)
        csv_path = os.path.join(Vars.base_path, "voltmap_local.csv")
        print(f"[DATA] DB failed or empty. Trying CSV: {csv_path}")
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # CSV 로드 시 시간 컬럼 처리 (필요시)
                if "CRTTIME" in df.columns:
                    df["CRTTIME"] = pd.to_datetime(df["CRTTIME"])
                print("[DATA] CSV loaded successfully.")
            except Exception as e:
                print(f"[DATA ERROR] CSV read failed: {e}")
                return None
        else:
            print(f"[DATA ERROR] CSV file not found at: {csv_path}")
            return 

    Vars.df_raw = df



def get_latest_time_from_db():
    query = """select * from STNVOLTMAP where STNLATITUDE !=0 and STNLONGITUDE !=0 and STNMAXVOLT !=0 and STNMINVOLT !=0 and STNAVGVOLT !=0 and STNNOMINALVOLT !=0 order by STNIDX desc fetch first 1 rows only"""
    payload = {
        "params": {"host": Vars.host, "port": Vars.port, "service_name": Vars.name, "user": Vars.user, "password": Vars.password, "dsn": f"{Vars.host}:{Vars.port}/{Vars.name}"},
        "query": query,
    }
    try:
        resp = requests.post(Vars.base_url + "receive_data_elx", json=payload, timeout=5)
        print('request complete _single data')
        df = pd.DataFrame(resp.json())

        tmp = df["CRTTIME"].iloc[0].replace(" GMT","")
        tmp = datetime.strptime(tmp, "%a, %d %b %Y %H:%M:%S")

        return tmp, resp.status_code

    except Exception as e:
        # 접속 거부(10061) 등 발생 시 그냥 None 리턴하여 에러 로그 폭발 방지
        return None




# =========================================================
# Kriging + Plotly Figure 생성
# =========================================================

#old 
#def add_geojson_boundary(fig, geojson_path, line_color="lightgray"):
#    gdf = gpd.read_file(geojson_path)
#
#    gdf = gdf[gdf.geometry.notnull()]
#    gdf = gdf[gdf.geometry.is_empty == False]
#    gdf = gdf[gdf.is_valid]
#
#    for geom in gdf.geometry:
#        if geom.geom_type == "Polygon":
#            x, y = geom.exterior.coords.xy
#            fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines", line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))
#
#        elif geom.geom_type == "MultiPolygon":
#            for poly in geom.geoms:
#                x, y = poly.exterior.coords.xy
#                fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines", line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False,))
#

def add_geojson_boundary(fig, geojson_path, line_color="white"):
    gdf = gpd.read_file(geojson_path)
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty & gdf.is_valid]

    # 모든 좌표를 담을 리스트
    all_x = []
    all_y = []

    for geom in gdf.geometry:
        # Polygon이든 MultiPolygon이든 반복문으로 처리
        polys = [geom] if geom.geom_type == "Polygon" else geom.geoms
        for poly in polys:
            x, y = poly.exterior.coords.xy
            all_x.extend(list(x))
            all_y.extend(list(y))
            # 중요: 도형 하나 끝날 때마다 None을 넣어줘야 선이 겹치지 않고 끊깁니다.
            all_x.append(None)
            all_y.append(None)

    # 단 한 번의 add_trace로 모든 경계선을 다 그립니다.
    fig.add_trace(go.Scatter(
        x=all_x, y=all_y, 
        mode="lines", 
        line=dict(color=line_color, width=1), 
        hoverinfo="skip", 
        showlegend=False
    ))

def build_voltmap_figure():
    if Vars.df_raw is None or Vars.df_raw.empty:
        fig = go.Figure()
        fig2 = go.Figure()
        fig.update_layout(template="plotly_dark", title="No Data Available")
        fig2.update_layout(template="plotly_dark", title="No Data Available")
        return fig, fig2

    # Vars에서 직접 파라미터 추출
    k_cfg = Vars.kriging_cfg
    v_cfg = Vars.voltage_cfg
    df = Vars.df_raw.copy()

    # 콤보박스 값(최대/최소/평균)에 따른 컬럼 결정 (사용자 exe 로직 스타일)
    if Vars.rep_flag == '최대':
        rep_col = 'STNMAXVOLT' if Vars.current_source == 'SE' else 'STNMAXVOLT_MEA'
    elif Vars.rep_flag == '최소':
        rep_col = 'STNMINVOLT' if Vars.current_source == 'SE' else 'STNMINVOLT_MEA'
    elif Vars.rep_flag == '평균':
        rep_col = 'STNAVGVOLT' if Vars.current_source == 'SE' else 'STNAVGVOLT_MEA'
    else:
        # 이리로 들어오면 안 됨 (디버깅용)
        print(f"Unknown RepFlag: {Vars.rep_flag}")
        return go.Figure()

    # 0값 제거 
    df = df[df[rep_col] != 0].copy()

    # Nominal Voltage 필터링 (765+345 케이스 포함)
    if Vars.nominal_flag == '765kV+345kV':
        df = df[df['STNNOMINALVOLT'].isin([765, 345])]
    elif Vars.nominal_flag in ['765kV', '345kV', '154kV']:
        # '765kV' -> 765
        v_int = int(Vars.nominal_flag.replace('kV', ''))
        df = df[df['STNNOMINALVOLT'] == v_int]
    # 'ALL' 이면 필터링 없이 통과

    # 4. 단위 변환 (kV 요청 시)
    if Vars.unit_flag == 'kV':
        df[rep_col] = df[rep_col] * df['STNNOMINALVOLT']

    # --- 이후 데이터 분리 및 Kriging (전달받은 rep_col 사용) ---
    devider = 34.32
    df_mainland = df[df["STNLATITUDE"] >= devider].copy()
    df_jeju = df[df["STNLATITUDE"] < devider].copy()

    v_min, v_norm, v_max = v_cfg["graph_min_value"], v_cfg["graph_normal_value"], v_cfg["graph_max_value"]
    norm_pos = (v_norm - v_min) / (v_max - v_min) if (v_max - v_min) != 0 else 0.5
    my_colorscale = [[0.0, v_cfg["graph_min_color"]], [norm_pos, v_cfg["graph_normal_color"]],[1.0, v_cfg["graph_max_color"]]]


    # 모델별 파라미터 설정
    variogram_params = None
    if k_cfg['model'] == 'linear':
        variogram_params = [k_cfg['slope'], k_cfg['nugget']]
    elif k_cfg['model'] in ['gaussian', 'spherical', 'exponential']:
        variogram_params = [k_cfg['range'], k_cfg['sill'], k_cfg['nugget']]
    elif k_cfg['model'] == 'power':
        variogram_params = [k_cfg['scale'], k_cfg['exponent'], k_cfg['nugget']]

    fig_land = go.Figure()
    fig_jeju = go.Figure()
    # 1. 배경 경계선 로드 
    add_geojson_boundary(fig_land, os.path.join(Vars.base_path, "shp.geojson"), line_color="#555")
    add_geojson_boundary(fig_jeju, os.path.join(Vars.base_path, "jeju.geojson"), line_color="#555")

    # 2. 본토(Mainland) Kriging
    lons = df_mainland["STNLONGITUDE"].values
    lats = df_mainland["STNLATITUDE"].values
    values = df_mainland[rep_col].values

    grid_lon = np.linspace(Vars.LAND_XMIN, Vars.LAND_XMAX, Vars.GRID_RESOL)
    grid_lat = np.linspace(Vars.LAND_YMIN, Vars.LAND_YMAX, Vars.GRID_RESOL)

    OK = OrdinaryKriging(lons, lats, values, variogram_model=k_cfg['model'], variogram_parameters=variogram_params, verbose=False)
    grid_z, _ = OK.execute("grid", grid_lon, grid_lat)
    gx, gy = np.meshgrid(grid_lon, grid_lat)

    # 본토 마스킹 (optimized.geojson 사용)
    land_outline = os.path.join(Vars.base_path, 'optimized.geojson')
    gdf_land = gpd.read_file(land_outline)
    boundary_land = unary_union(list(gdf_land.geometry))
    mask = vectorized.contains(boundary_land, gx, gy)
    grid_z[~mask] = np.nan
    tooltips = f'''
        <b>전압 정보</b><br>
        경도: %{{x:.2f}}<br>
        위도: %{{y:.2f}}<br>
        값: %{{z:.3f}} {Vars.unit_flag}<br>
        <extra></extra>
    '''

    # 본토 Contour 추가
    fig_land.add_trace(go.Contour(x=grid_lon, y=grid_lat, z=grid_z, colorscale=my_colorscale, opacity=0.65, contours=dict(showlines=False), showscale=False,
        hovertemplate=tooltips

    ))

    # -----------------------------------------------------
    # 3. 제주(Jeju) Kriging (xaxis2, yaxis2 사용)
    # -----------------------------------------------------
    lons_j = df_jeju["STNLONGITUDE"].values
    lats_j = df_jeju["STNLATITUDE"].values
    values_j = df_jeju[rep_col].values


    grid_lon_j = np.linspace(Vars.JEJU_XMIN, Vars.JEJU_XMAX, Vars.GRID_RESOL)
    grid_lat_j = np.linspace(Vars.JEJU_YMIN, Vars.JEJU_YMAX, Vars.GRID_RESOL)
    
    OK_j = OrdinaryKriging(lons_j, lats_j, values_j, variogram_model=k_cfg['model'], variogram_parameters=variogram_params, verbose=False)
    grid_z_j, _ = OK_j.execute("grid", grid_lon_j, grid_lat_j) 
    gx, gy = np.meshgrid(grid_lon_j, grid_lat_j)

    # 제주 마스킹
    jeju_geo_path = os.path.join(Vars.base_path, 'jeju.geojson')
    gdf_jeju_poly = gpd.read_file(jeju_geo_path)
    boundary_jeju = unary_union(list(gdf_jeju_poly.geometry))
    mask_j = vectorized.contains(boundary_jeju, gx, gy)
    grid_z_j[~mask_j] = np.nan


    # 제주 Contour 
    fig_jeju.add_trace(go.Contour(x=grid_lon_j, y=grid_lat_j, z=grid_z_j, colorscale=my_colorscale, opacity=0.65, contours=dict(showlines=False), showscale=False))

    # 4. 변전소 마커 추가
    fig_land.add_trace(go.Scatter(x=df_mainland["STNLONGITUDE"], y=df_mainland["STNLATITUDE"], mode="markers", marker=dict(size=3, color="black"), hoverinfo="skip", showlegend=False))
    
    # 제주 변전소 마커 
    fig_jeju.add_trace(go.Scatter(x=df_jeju["STNLONGITUDE"], y=df_jeju["STNLATITUDE"], mode="markers", marker=dict(size=3, color="black"), hoverinfo="skip", showlegend=False))

    # 공통 레이아웃 설정 함수 (중복 제거용)
    def apply_common_layout(fig, x_range, y_range):
        fig.update_layout(
            autosize=True,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#15191c",
            plot_bgcolor="#15191c",
            xaxis=dict(
                range=x_range,
                visible=False,
                # [추가] 오토 스케일과 유사한 느낌을 주도록 여백 강제 제거
                constrain="domain", 
                autorange=False      # Plotly가 스스로 계산하지 못하게 함
            ),
            yaxis=dict(
                range=y_range,
                visible=False,
                scaleanchor="x",     # 가로세로 비율 유지 (지도가 찌그러지지 않게 함)
                scaleratio=1,
                autorange=False
            ),
            showlegend=False
        )


    # 본토 레이아웃 적용 (기존 Vars 범위 사용)
    apply_common_layout(fig_land, [Vars.LAND_XMIN, Vars.LAND_XMAX], [Vars.LAND_YMIN, Vars.LAND_YMAX])

    # 제주도 레이아웃 적용 (제주도 좌표에 집중)
    apply_common_layout(fig_jeju, [Vars.JEJU_XMIN, Vars.JEJU_XMAX], [Vars.JEJU_YMIN, Vars.JEJU_YMAX])

    return fig_land, fig_jeju

def build_colorbar_figure():
    v_cfg = Vars.voltage_cfg
    
    # 1. 절대값 가져오기 (PU든 kV든 설정된 값 그대로)
    v_min = v_cfg["graph_min_value"]
    v_norm = v_cfg["graph_normal_value"]
    v_max = v_cfg["graph_max_value"]
    
    # 2. 범위 체크 (분모가 0이 되는 것 방지)
    total_range = v_max - v_min
    if total_range <= 0:
        total_range = 1 # 에러 방지용 임시값

    # 3. 가운데(Normal)의 상대적 위치 계산 (0.0 ~ 1.0 사이)
    norm_pos = (v_norm - v_min) / total_range
    
    # 4. 범위를 벗어나지 않게 클램핑 (안전장치)
    norm_pos = max(0.01, min(0.99, norm_pos))

    # 5. 사용자 정의 컬러스케일 구성
    my_colorscale = [
        [0.0, v_cfg["graph_min_color"]],      # 양 끝 (최저)
        [norm_pos, v_cfg["graph_normal_color"]], # 사용자가 지정한 가운데 값
        [1.0, v_cfg["graph_max_color"]]       # 양 끝 (최고)
    ]

    fig = go.Figure(go.Heatmap(
        z=[None, None],       # 데이터는 넣어줘야 컬러바가 생성됩니다.
        colorscale=my_colorscale,
        showscale=True,
        opacity=0,                # [핵심] 배경 히트맵을 투명하게 숨깁니다.
        hoverinfo="skip",         # 배경에 마우스 올려도 반응 없게 함
        colorbar=dict(
            orientation="h",
            thickness=18,
            len=0.9,
            x=0.5, y=0.5,
            xanchor="center", yanchor="middle",
            tickvals=[v_min, v_norm, v_max],
            tickformat=".3f",
            tickfont=dict(color="white", size=10),
            outlinecolor="rgba(0,0,0,0)", # 컬러바 외곽선 제거
            ypad=0,               # 패딩 제거하여 내부 공간 극대화
        ),
    ))
    
    # 레이아웃은 이전과 동일
    fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=5), # 여백 최소화
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="#15191c", 
        plot_bgcolor="#15191c", 
        # [수정] 고정 height를 지우거나 vh 단위로 맞춤
        height=None,           # 부모 Div의 높이에 맞게 자동 조절되도록 유도
        autosize=True,         # 컨테이너 크기에 맞춤
        width=400,             # 너비는 가급적 유지 (제목/눈금 겹침 방지)


    )
    return fig


# =========================================================
# Dash App
# =========================================================
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], assets_folder=os.path.join(Vars.base_path, "assets")) # 명시적으로 지정!

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


initial_time = "대기 중"
if Vars.df_raw is not None and not Vars.df_raw.empty:
    try:
        # 데이터가 있으면 그 시간으로 초기값 설정
        initial_time = Vars.df_raw["CRTTIME"].iloc[0].strftime("%Y-%m-%d %H:%M:%S")
    except:
        pass

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
                        options=[{"label": "자동 갱신", "value": 1}], value=[1], id="auto-update-check",
                        className="custom-check", inline=True, style={"display": "flex", "alignItems": "center"}
                    )
                ], width="auto", className="me-3"),


                dbc.Col([dbc.Button("설정", id="open-settings", className="custom-btn me-2"),], width="auto")
            ], className="header-container px-3 py-2 align-items-center"),


    # 1초마다 스레드 상태를 체크할 타이머 (Qt의 Timer 역할)
    dcc.Interval(id="timer-checker", interval=2000, n_intervals=0),
    # 레이아웃에 표시할 초기 시간 결정


    html.Div([
        # [수정] 수행 시각과 컬러바를 한 줄에 배치
        dbc.Row([
            # 왼쪽: 수행 시각 라벨
            dbc.Col([
                html.Span("수행 시각 : ", style={"fontSize": "1.6vh", "color": "#bbb"}),
                html.Span(initial_time, id="time-display", className="time-text"), 
                html.Span("(SCADA)", className="ms-2 opacity-50", 
                          style={"fontSize": "1.4vh",
                                 "backgroundColor": "#15191c", 
                                 "minHeight": "50px",   # 최소한 이 정도 높이는 유지 (모바일이나 작은 창 대비)
                                 "height": "6vh",       # 화면 높이의 6%를 차지하도록 설정
                                 "padding": "0 15px",
                                 "display": "flex",
                                 "alignItems": "center"
                                 
                                 })
            ], width=True, className="d-flex align-items-center"),

            # 오른쪽: 컬러바(범례) 영역
            dbc.Col([
                html.Div(id="color-bar-legend", className="color-bar-container")
            ], width="auto", className="d-flex align-items-center justify-content-end")
        ], className="mb-2 align-items-center", style={"backgroundColor": "#15191c"}),

        # 메인 캔버스
        html.Div(id="main-graph-canvas", className="canvas-border", style={"height": "70vh",  "backgroundColor": "#15191c"}),
        
        # 하단 서브 캔버스
        html.Div(id="sub-graph-canvas", className="canvas-border mt-2", style={"height": "18vh",  "width":'50vw', "backgroundColor": "#15191c"})
        
    ], className="main-content m-2"),

    # 전역 로딩 오버레이 (GIF 버전)
    html.Div(
        id="global-loading-overlay",
        children=[
            html.Div([
                html.Img(src="/assets/load_line.gif", style={"width": "250px", "height": "auto"}),
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
    html.Div(id="terminator")  

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
    global THREAD_STATE
    print('run update process')
    # 이미 실행 중이면 중복 실행 방지
    if THREAD_STATE["is_running"] or THREAD_STATE["status"] == "running": return

    # 진입하자마자 상태부터 변경해서 다음 콜백이 못 들어오게 막음
    THREAD_STATE["status"] = "running"
    THREAD_STATE["is_running"] = True

    thread = threading.Thread(target=data_update_and_build_worker)
    thread.daemon = True
    thread.start()
    print('thread start')


def data_update_and_build_worker():
    global THREAD_STATE
    print("!!!! WORKER STARTED !!!!")

    try:
        THREAD_STATE["status"] = "running"
        THREAD_STATE["is_running"] = True

        get_all_data_from_oracle()
        print('data updated')

        fig_land, fig_jeju = build_voltmap_figure()
        colorbar = build_colorbar_figure()
        # [4] 결과 저장
        THREAD_STATE["fig"] = (fig_land, fig_jeju, colorbar)
        THREAD_STATE["status"] = "done"
        print('figure updated')
        print("[THREAD] 그래프 생성 완료.")
    
        print("!!!! WORKER FINISHED !!!!")

    except Exception as e:
        print(f"[THREAD CRITICAL ERROR] {e}")
        THREAD_STATE["status"] = "error"
    finally:
        THREAD_STATE["is_running"] = False



@app.callback(
    [
        Output("main-graph-canvas", "children"), Output("sub-graph-canvas", "children"), Output("color-bar-legend", "children"),
        Output("time-display", "children"), Output("global-loading-overlay", "style", allow_duplicate=True)
    ],
    Input("timer-checker", "n_intervals"),
    [State("auto-update-check", "value"), State("time-display", "children"), State("config-store", "data")],
    prevent_initial_call='initial_duplicate'
)
def monitor_system(n, auto_update_val, current_display_time, config):
    global THREAD_STATE
    print(THREAD_STATE["status"])


    # [Case 1] 백그라운드 작업이 완료되었는지 확인 (Redraw)
    if THREAD_STATE["status"] == "done":
        THREAD_STATE["status"] = "idle"
        print(f"[UI] Done state detected. Data Time: {current_display_time} -> Updating UI")
        fig_land, fig_jeju, fig_colorbar = THREAD_STATE["fig"]

        data_time = "No Data Time"
        if Vars.df_raw is not None and not Vars.df_raw.empty:
            # CRTTIME 컬럼의 첫 번째 값(최신값)을 가져와서 포맷팅
            try:
                data_time = Vars.df_raw["CRTTIME"].iloc[0].strftime("%Y-%m-%d %H:%M:%S")

            except Exception as e:
                data_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")



        return [
                    dcc.Graph(figure=fig_land, style={"height": "100%", "width": "100%"}),    # main-graph-canvas
                    dcc.Graph(figure=fig_jeju, style={"height": "100%", "width": "100%"}),    # sub-graph-canvas
                    dcc.Graph(figure=fig_colorbar, config={'displayModeBar': False}, style={"height": "50%", "width": "100%"}),  # [확인용] 노란색 외곽선 추가
            # color-bar-legend
                    data_time,                                                               # time-display
                    {"display": "none"}                                                      # 로딩바 숨김
                ]

    # [Case 2] 자동 갱신이 켜져 있고, 실행 중이 아닐 때 DB 체크
    if auto_update_val and not THREAD_STATE["is_running"]:
        latest_time = get_latest_time_from_db()
        
        # 시간이 달라졌다면 (DB 업데이트 발생)
        if latest_time and str(latest_time) != str(current_display_time):
            run_update_process(config) # 스레드 시작

            overlay_style = {
                    "position": "fixed", "top": 0, "left": 0, "width": "100vw", "height": "100vh",
                    "backgroundColor": "rgba(0, 0, 0, 0.8)", "display": "flex",
                    "alignItems": "center", "justifyContent": "center", "zIndex": 9999
                }
            return [no_update, no_update,no_update, no_update, overlay_style] # 로딩바 표시

    return [no_update] * 5




# 창이 닫힐 때 서버의 /shutdown 경로로 신호를 보냄
clientside_callback(
    """
    function(id) {
        window.addEventListener('beforeunload', function (e) {
            navigator.sendBeacon('/shutdown'); 
        });
        return "";
    }
    """,
    Output("terminator", "children"),
    Input("terminator", "id")
)

# 서버를 종료시키는 라우트
@app.server.route('/shutdown', methods=['POST'])
def shutdown():
    print("브라우저 종료 감지: 프로세스를 종료합니다...")
    os._exit(0) # 프로세스 자체를 즉시 파괴
    return "Server shutting down..."



# =========================================================
# Run Server
# =========================================================
if __name__ == "__main__":
    # 1. 서버 시작과 동시에 백그라운드 연산 스레드 딱 하나만 실행
    load_config()

    run_update_process(None) 
    
    # 2. 브라우저 자동 열기
    Timer(1, open_browser).start()
    
    # 3. 대시 서버 실행
    app.run_server(host="127.0.0.1", port=8050, debug=False) # 디버그 모드 시 스레드 2배 생성 방지




