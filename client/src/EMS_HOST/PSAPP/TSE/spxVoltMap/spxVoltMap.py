# -*- coding: cp949 -*-
# base imports
from tabnanny import check
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
from dash import Dash, html, Input, Output, clientside_callback, dcc, State, no_update
import plotly.graph_objects as go

from cryptography.fernet import Fernet
import keyring

import os, sys, configparser, time
from io import StringIO
import pysftp

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
# # key = generate_new_key()
# key = b'3vqHCJxl_vDg-QA2L2ZilVUnObnsw__JvCE7ZWIKK3E='
# 
# enc_id = encrypt_text('ems', key)
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
    current_source = 'SE' # or MEA
        
    # 3. 리소스 및 기타 설정
    name_ref = None
    voltage_cfg = {}
    kriging_cfg = {}
        
    # 4. 경로 설정 (전역 유지)
    base_path = os.getcwd().replace("\\", "/")
    sysconf_path = os.path.join(base_path, '../project/EMS/conf/sysconf.ini').replace("\\", "/")
    base_path = os.path.join(base_path, 'HistViewer/spxVoltMap/_internal/').replace("\\", "/")


    # Constants
    LAND_XMIN, LAND_XMAX = 125.5, 129.8
    LAND_YMIN, LAND_YMAX = 34.2, 38.8

    JEJU_XMIN, JEJU_XMAX = 126, 127
    JEJU_YMIN, JEJU_YMAX = 33.2, 33.6
    
    devider = 34.32


    mapping_dict = None

    LAND_BOUNDARY_TRACE = None
    JEJU_BOUNDARY_TRACE = None

    loading_on = {
        "position": "fixed",
        "top": "50%",      # 화면 위에서 50% 지점
        "left": "50%",     # 화면 왼쪽에서 50% 지점
        "transform": "translate(-50%, -50%)", # 본인 크기의 절반만큼 역이동하여 완벽한 중앙 정렬
        "width": "300px",   # 스피너가 들어갈 박스 크기만 설정
        "height": "200px",
        "backgroundColor": "rgba(0, 0, 0, 0.8)",
        "borderRadius": "15px", # 모서리를 둥글게 하면 더 예쁩니다.
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "zIndex": 9999,
        "flexDirection": "column",
        "color": "white"
    }

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

def check_thread_context(label=""):
    """현재 실행 중인 스레드가 메인 스레드인지 확인하고 정보를 출력합니다."""
    curr_thread = threading.current_thread()
    main_thread = threading.main_thread()
    
    is_main = curr_thread == main_thread
    thread_name = curr_thread.name
    thread_id = threading.get_ident()
    
    status = "MAIN THREAD" if is_main else "SUB THREAD (Background)"
    
    print(f"[{label}] --------------------------------")
    print(f"  - Status: {status}")
    print(f"  - Thread Name: {thread_name}")
    print(f"  - Thread ID: {thread_id}")
    print(f"------------------------------------------")
    
    return is_main

# exe 가 바로 웹 열게 하는 부분
def open_browser():
    import webbrowser
    webbrowser.open("http://127.0.0.1:8050")

def load_config():
    #check_thread_context('load config')
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

    stn_username = cipher.decrypt(config['SS_name_ref']['username'].encode()).decode()
    stn_password = cipher.decrypt(config['SS_name_ref']['pw'].encode()).decode()
    stn_csv_path = config['SS_name_ref']['stn_csv_path']
    stn_port = int(config['SS_name_ref']['port'])


    # voltage boundary
    vb = config['voltage_boundary']
    Vars.voltage_cfg = {
        "graph_ov_color": vb['graph_ov_color'].strip('"'),
        "graph_max_color": vb['graph_max_color'].strip('"'),
        "graph_max_value": float(vb['graph_max_value']),
        "graph_normal_color": vb['graph_normal_color'].strip('"'),
        "graph_normal_value": float(vb['graph_normal_value']),
        "graph_min_color": vb['graph_min_color'].strip('"'),
        "graph_min_value": float(vb['graph_min_value']),
        "graph_uv_color": vb['graph_uv_color'].strip('"'),
        "level_density_high": int(vb['level_density_high']),
        "level_density_low": int(vb['level_density_low']),
        "level_density": int(vb['level_density']),
        "point_max_color": vb['point_max_color'].strip('"'),
        "point_normal_color": vb['point_normal_color'].strip('"'),
        "point_min_color": vb['point_min_color'].strip('"'),
        "point_edge_width": float(vb['point_edge_width']),
        "point_edge_color": vb['point_edge_color'].strip('"'),
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



    config = configparser.ConfigParser()

    try :
        with open(Vars.sysconf_path, 'r',encoding='cp949') as file:
            config.read_file(file)
            
            NodeGroup = config['LOCAL']['NodeGroup'].split('_')[-1]
            host = config[f'{NodeGroup}']['Host01A']
    

            cnopts = pysftp.CnOpts()
            cnopts.hostkeys = None

            try:
                #with pysftp.Connection(host=host, username=stn_username, password=stn_password, port=stn_port, cnopts=cnopts) as sftp:
                #    # read remote binary
                #    with sftp.open(stn_csv_path, mode='rb') as remote_file:
                #        raw_data = remote_file.read()
                #
                #    # detect encoding
                #    encodings_to_try = ['cp949','utf-8', 'euc-kr', 'latin-1']
                #    decoded_data = None
                #
                #    for encoding in encodings_to_try:
                #        try:
                #            decoded_data = raw_data.decode(encoding)
                #            break
                #
                #        except UnicodeDecodeError:
                #            print(f' UnicodeDecodeError: {encoding} fail')
                #
                #    if decoded_data is not None:
                #        csv_content = pd.read_csv(StringIO(decoded_data))
                #        # for index, row in csv_content.iterrows():
                #            # print(index, row.to_dict(), row.to_dict())
                
                stn_csv_path = os.path.join(Vars.base_path, 'Station.csv').replace("\\", "/")
                
                with open(stn_csv_path, 'rb') as f:
                    raw_data = f.read()
                
                
                    # detect encoding
                    encodings_to_try = ['cp949','utf-8', 'euc-kr', 'latin-1']
                    decoded_data = None
                
                    for encoding in encodings_to_try:
                        try:
                            decoded_data = raw_data.decode(encoding)
                            break
                
                        except UnicodeDecodeError:
                            print(f' UnicodeDecodeError: {encoding} fail')
                
                    if decoded_data is not None:
                        csv_content = pd.read_csv(StringIO(decoded_data))
                
                        name_ref = csv_content[['Unnamed: 1', 'Unnamed: 2']].dropna().copy()
                        Vars.mapping_dict = name_ref.set_index('Unnamed: 1')['Unnamed: 2'].to_dict()


            except Exception as e:
                    print(f'[PYSFTP] Exception : {e}')

    except Exception as e:
            print(f'[CONFIG PARSER] : Exception : {e}')

    ########### ini                     


load_config()
# =========================================================
# Flask-Oracle API (without Qt)
# =========================================================

def get_all_data_from_oracle(source="SE"):
    #check_thread_context('get all data from oracle')
    if source == "SE":
        query = """select * from STNVOLTMAP where STNLATITUDE != 0 and STNLONGITUDE != 0 and STNNOMINALVOLT != 0 and STNMAXVOLT != 0 and STNMINVOLT != 0 and STNAVGVOLT != 0 order by CRTTIME desc """

    else:
        query = """select * from STNVOLTMAP where STNLATITUDE != 0 and STNLONGITUDE != 0 and STNNOMINALVOLT != 0 and STNMAXVOLT_MEA != 0 and STNMINVOLT_MEA != 0 and STNAVGVOLT_MEA != 0 order by CRTTIME desc"""

    payload = {
        "params": {"host": Vars.host, "port": Vars.port, "service_name": Vars.name, "user": Vars.user, "password": Vars.password, "dsn": f"{Vars.host}:{Vars.port}/{Vars.name}"},
        "query": query,
    }

    df = None # 초기값을 None으로 설정하여 모호함 제거

    try:    
        plus_url = 'receive_data_elx'
        dsn = f'{Vars.base_url}'
        resp = requests.post(dsn + plus_url, json=payload)
        df = pd.DataFrame(resp.json())
        if not df.empty: df["CRTTIME"] = df['CRTTIME'].apply(lambda x: datetime.strptime(x.replace(' GMT', ''), "%a, %d %b %Y %H:%M:%S") if isinstance(x, str) else None)


    except Exception as e:
            print(f"[DB ERROR] : {e}")
            if Vars.df_raw is not None:
                return


    if df is None or (isinstance(df, pd.DataFrame) and df.empty):        
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

    df['stn_name'] = df['STNRIDX'].map(lambda x: Vars.mapping_dict.get(x, 'Unknown'))
    Vars.df_raw = df
    print('[DATA] Complete Data update')

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

def make_geojson_boundary(geojson_path, line_color="white"):
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
    return go.Scatter(x=all_x, y=all_y, mode="lines", line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False)

def build_voltmap_figure():
    #check_thread_context('build voltmap figure')
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
    devider = Vars.devider
    df_mainland = df[df["STNLATITUDE"] >= devider].copy()
    df_jeju = df[df["STNLATITUDE"] < devider].copy()

    my_colorscale = [
        [0.0, v_cfg["graph_uv_color"]], [0.2, v_cfg["graph_min_color"]], [0.5, v_cfg["graph_normal_color"]], [0.8, v_cfg["graph_max_color"]], [1.0, v_cfg["graph_ov_color"]]]
        # 0.90 미만 (매우 위험)
        # 0.95 (경고 시작)
        # 1.00 (가장 평온한 상태)
        # 1.05 (경고 시작)
        # 1.10 이상 (매우 위험)
    



    # 모델별 파라미터 설정
    variogram_params = None
    if k_cfg['model'] == 'linear':
        variogram_params = [k_cfg['slope'], k_cfg['nugget']]
    elif k_cfg['model'] in ['gaussian', 'spherical', 'exponential']:
        variogram_params = [k_cfg['range'], k_cfg['sill'], k_cfg['nugget']]
    elif k_cfg['model'] == 'power':
        variogram_params = [k_cfg['scale'], k_cfg['exponent'], k_cfg['nugget']]

    fig_land = go.Figure()

    if Vars.LAND_BOUNDARY_TRACE is None: Vars.LAND_BOUNDARY_TRACE =make_geojson_boundary(os.path.join(Vars.base_path, "shp.geojson"), line_color="#555")
    fig_land.add_trace(Vars.LAND_BOUNDARY_TRACE)


    # 2. 본토(Mainland) Kriging
    lons = df_mainland["STNLONGITUDE"].values
    lats = df_mainland["STNLATITUDE"].values
    values = df_mainland[rep_col].values

    grid_lon = np.linspace(Vars.LAND_XMIN, Vars.LAND_XMAX, v_cfg['level_density'])
    grid_lat = np.linspace(Vars.LAND_YMIN, Vars.LAND_YMAX, v_cfg['level_density'])

    OK = OrdinaryKriging(lons, lats, values, variogram_model=k_cfg['model'], variogram_parameters=variogram_params, verbose=False)
    grid_z, _ = OK.execute("grid", grid_lon, grid_lat)
    gx, gy = np.meshgrid(grid_lon, grid_lat)


    # 본토 마스킹 (optimized.geojson 사용)
    land_outline = os.path.join(Vars.base_path, 'optimized.geojson')
    gdf_land = gpd.read_file(land_outline)
    boundary_land = unary_union(list(gdf_land.geometry))
    mask = vectorized.contains(boundary_land, gx, gy)
    grid_z[~mask] = np.nan


    tooltips = (
        "<span style='font-size: 1.1em; font-weight: bold;'>%{customdata}</span><br>" +
        "전압: <b>%{text:.3f}</b> " + f"{Vars.unit_flag}<br>" +
        "<extra></extra>"
    )
    
    # 본토 Contour 추가
    fig_land.add_trace(go.Contour(x=grid_lon, y=grid_lat, z=grid_z, colorscale=my_colorscale, opacity=0.65, contours=dict(showlines=False), showscale=False, hoverinfo="skip",))

    # 본토 변전소 마커 추가
    fig_land.add_trace(go.Scatter(x=lons, y=lats, mode="markers", text=values, customdata=df_mainland['stn_name'], hovertemplate=tooltips,
                                  marker=dict(size=v_cfg['point_size'], color=v_cfg['point_normal_color'],line=dict(width=v_cfg['point_edge_width'], color=v_cfg['point_edge_color']))))



    # 공통 레이아웃 설정 함수 (중복 제거용)
    def apply_common_layout(fig, x_range, y_range):
        fig.update_layout(
            autosize=True,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#15191c",
            plot_bgcolor="#15191c",
            hoverlabel=dict(
                bgcolor="#2d3035",      # 조금 더 부드러운 다크 그레이
                bordercolor="#5dade2",  # 포인트 컬러(하늘색)로 테두리 강조
                font=dict(size=14, color="#ffffff", family="Apple SD Gothic Neo, Malgun Gothic"), # 폰트 가독성
                align="left",
                namelength=-1           # 이름 짤림 방지
            ),
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



    # -----------------------------------------------------
    # 3. 제주(Jeju) Kriging (xaxis2, yaxis2 사용)
    # -----------------------------------------------------
    fig_jeju = go.Figure()

    # 1. 배경 경계선 로드 
    if Vars.JEJU_BOUNDARY_TRACE is None: Vars.JEJU_BOUNDARY_TRACE = make_geojson_boundary(os.path.join(Vars.base_path, "jeju.geojson"), line_color="#555")
    fig_jeju.add_trace(Vars.JEJU_BOUNDARY_TRACE)


    if not df_jeju .empty:
        lons_j = df_jeju["STNLONGITUDE"].values
        lats_j = df_jeju["STNLATITUDE"].values
        values_j = df_jeju[rep_col].values


        grid_lon_j = np.linspace(Vars.JEJU_XMIN, Vars.JEJU_XMAX, v_cfg['level_density'])
        grid_lat_j = np.linspace(Vars.JEJU_YMIN, Vars.JEJU_YMAX, v_cfg['level_density'])
    
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
        fig_jeju.add_trace(go.Contour(x=grid_lon_j, y=grid_lat_j, z=grid_z_j, colorscale=my_colorscale, opacity=0.65, contours=dict(showlines=False), showscale=False, hoverinfo="skip",))
    
        # 제주 변전소 마커 
        fig_jeju.add_trace(go.Scatter(x=lons_j, y=lats_j, mode="markers", text=values_j, customdata=df_jeju['stn_name'], hovertemplate=tooltips,
                                      marker=dict(size=v_cfg['point_size'], color=v_cfg['point_normal_color'],line=dict(width=v_cfg['point_edge_width'], color=v_cfg['point_edge_color']))))


    

    # 제주도 레이아웃 적용 (제주도 좌표에 집중)
    apply_common_layout(fig_jeju, [Vars.JEJU_XMIN, Vars.JEJU_XMAX], [Vars.JEJU_YMIN, Vars.JEJU_YMAX])

    return fig_land, fig_jeju


def build_colorbar_figure():
    v_cfg = Vars.voltage_cfg

    # 이 값들은 오직 '위치'를 잡는 데만 쓰입니다.
    p_min = v_cfg["graph_min_value"]   # 예: 0.90
    p_norm = v_cfg["graph_normal_value"] # 예: 1.00
    p_max = v_cfg["graph_max_value"]   # 예: 1.08



    # 1. 공칭전압(Multiplier) 추출 로직
    import re
    try:
        # Vars.nominal_flag가 "345kV"라면 345를 추출, "ALL"이면 1.0 (또는 기준값)
        nom_val_match = re.search(r'\d+', str(Vars.nominal_flag))
        multiplier = float(nom_val_match.group()) if nom_val_match else 1.0
    except:
        multiplier = 1.0
    
        
    m = multiplier if Vars.unit_flag == "kV" else 1.0
    t_min, t_norm, t_max = f"{p_min*m:.2f}", f"{p_norm*m:.2f}", f"{p_max*m:.2f}"
    
    # 색상 위치 계산 (PU 기준)
    total = p_max - p_min if p_max > p_min else 1.0
    norm_pos = (p_norm - p_min) / total

    # [수정 3] 파랑(0~10%)과 빨강(90~100%)이 확실히 보이고 중앙은 초록이 오도록 배치
    my_colorscale = [
        [0.0, v_cfg["graph_uv_color"]],     # 파랑 시작
        [0.1, v_cfg["graph_uv_color"]],     # 파랑 유지
        [0.11, v_cfg["graph_min_color"]],   # 시안 시작
        [norm_pos, v_cfg["graph_normal_color"]], # 중앙 초록
        [0.89, v_cfg["graph_max_color"]],   # 노랑 끝
        [0.9, v_cfg["graph_ov_color"]],     # 빨강 시작
        [1.0, v_cfg["graph_ov_color"]]      # 빨강 끝
    ]

    # [수정 핵심] Heatmap 대신 Scatter를 사용해 본체(빨간 박스)를 아예 생성하지 않음
    fig = go.Figure(go.Scatter(
        x=[p_min, p_max], 
        y=[0, 0],
        mode="markers",
        marker=dict(
            size=0,           # 점 크기를 0으로 해서 투명하게 만듦
            color=[p_min, p_max],
            colorscale=my_colorscale,
            showscale=True,   # 컬러바 막대기만 빌려옴
            colorbar=dict(
                orientation="h",
                thickness=17,
                len=1.0,
                x=0.5, y=0.5, # 막대기 위치를 위로 밀어올림
                xanchor="center",
                tickmode="array",
                tickvals=[],
                outlinecolor="rgba(0,0,0,0)",
                borderwidth=0
            ),
        ),
        hoverinfo="skip"
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", # 이제 여기가 투명해서 빨간색이 안 보임
        margin=dict(l=30, r=30, t=5, b=35),
        xaxis=dict(
            visible=True,
            fixedrange=True,
            range=[p_min, p_max],
            tickmode="array",
            tickvals=[p_min, p_norm, p_max],
            ticktext=[t_min, t_norm, t_max],
            tickfont=dict(color="white", size=12),
            showgrid=False,
            showline=False, 
            zeroline=False,
            ticks="",
        ),
        yaxis=dict(visible=False, range=[-1, 1]),
        height=70,
        width=510,
    )

    return fig


def run_update_process(config):
    global THREAD_STATE
    # 이미 실행 중이면 중복 실행 방지
    if THREAD_STATE["is_running"] or THREAD_STATE["status"] == "running": return

    # 진입하자마자 상태부터 변경해서 다음 콜백이 못 들어오게 막음
    THREAD_STATE["status"] = "running"
    THREAD_STATE["is_running"] = True

    thread = threading.Thread(target=data_update_and_build_worker)
    thread.daemon = True
    thread.start()

def data_update_and_build_worker():
    global THREAD_STATE
    print("[THREAD] 동작 시작")

    try:
        THREAD_STATE["status"] = "running"
        THREAD_STATE["is_running"] = True

        get_all_data_from_oracle()


        fig_land, fig_jeju = build_voltmap_figure()
        colorbar = build_colorbar_figure()

        # [4] 결과 저장
        THREAD_STATE["fig"] = (fig_land, fig_jeju, colorbar)
        THREAD_STATE["status"] = "done"
        print("[THREAD] 그래프 생성 완료.")
    
    except Exception as e:
        print(f"[THREAD CRITICAL ERROR] {e}")
        THREAD_STATE["status"] = "error"
    finally:
        THREAD_STATE["is_running"] = False



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


        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("대표전압", className="fw-bold mb-1"),
                        dbc.Select(
                            id="rep-flag-select",
                            options=[
                                {"label": "최대", "value": "최대"},
                                {"label": "평균", "value": "평균"},
                                {"label": "최소", "value": "최소"},
                            ],
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.Label("정격전압", className="fw-bold mb-1"),
                        dbc.Select(
                            id="nominal-flag-select",
                            options=[
                                {"label": "ALL", "value": "ALL"},
                                {"label": "765kV", "value": "765kV"},
                                {"label": "345kV", "value": "345kV"},
                                {"label": "154kV", "value": "154kV"},
                                {"label": "765kV + 345kV", "value": "765kV+345kV"},
                            ],
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.Label("단위", className="fw-bold mb-1"),
                        dbc.Select(
                            id="unit-flag-select",
                            options=[
                                {"label": "PU", "value": "PU"},
                                {"label": "kV", "value": "kV"},
                            ],
                        ),
                    ],
                    width=4,
                ),
            ],
            className="mb-3",
        ),



        # 그래프 및 포인트 설정 (색상 및 수치)
        dbc.Row([
            # [1열]
            dbc.Col([
                html.Label("그래프 임계값 설정", className="fw-bold mb-2"),
                dbc.InputGroup([dbc.InputGroupText("최댓값"), dbc.Input(id="graph-max-val", type="number", value=Vars.voltage_cfg['graph_max_value'])], className="setting-label mb-2"),
                dbc.InputGroup([dbc.InputGroupText("중간값"), dbc.Input(id="graph-norm-val", type="number", value=Vars.voltage_cfg['graph_normal_value'])], className="setting-label mb-2"),
                dbc.InputGroup([dbc.InputGroupText("최솟값"), dbc.Input(id="graph-min-val", type="number", value=Vars.voltage_cfg['graph_min_value'])], className="setting-label mb-2"),
                dbc.InputGroup([dbc.InputGroupText("정밀도"), dbc.Input(id="grid-resol-input", type="number", value=Vars.voltage_cfg['level_density'])], className="setting-label mb-2"),
            ], width=4, className="pe-4"),


            # [2열] 색상 팔레트 설정 (우측 정렬 및 비율 유지)
            dbc.Col([
                html.Label("그래프 색상 설정", className="fw-bold mb-2"),
                
                # 가독성을 위해 개별적으로 작성 (Vars 값 직접 참조)
                html.Div([
                    html.Label("과전압", className="m-0"),
                    dcc.Input(type="color", id="color-ov", value=Vars.voltage_cfg['graph_ov_color'], className="custom-color-picker")
                ], className="d-flex align-items-center justify-content-between mb-2"),

                html.Div([
                    html.Label("최댓값", className="m-0"),
                    dcc.Input(type="color", id="color-g-max", value=Vars.voltage_cfg['graph_max_color'], className="custom-color-picker")
                ], className="d-flex align-items-center justify-content-between mb-2"),

                html.Div([
                    html.Label("중간값", className="m-0"),
                    dcc.Input(type="color", id="color-g-norm", value=Vars.voltage_cfg['graph_normal_color'], className="custom-color-picker")
                ], className="d-flex align-items-center justify-content-between mb-2"),

                html.Div([
                    html.Label("최솟값", className="m-0"),
                    dcc.Input(type="color", id="color-g-min", value=Vars.voltage_cfg['graph_min_color'], className="custom-color-picker")
                ], className="d-flex align-items-center justify-content-between mb-2"),

                html.Div([
                    html.Label("저전압", className="m-0"),
                    dcc.Input(type="color", id="color-uv", value=Vars.voltage_cfg['graph_uv_color'], className="custom-color-picker")
                ], className="d-flex align-items-center justify-content-between mb-2"),
            ], width=3, className="px-4"),


            # [3열] 포인트 설정
            dbc.Col([
                html.Label("포인트 설정", className="fw-bold mb-2"),
                html.Div([
                    html.Label("크기", className="m-0"),
                    dbc.Input(id="point-size", type="number", value=Vars.voltage_cfg['point_size'], size="sm", style={"flex": "0 0 50%"}),
                ], className="d-flex align-items-center justify-content-between mb-3"),
                
                html.Div([
                    html.Label("색상", className="m-0"),
                    dcc.Input(type="color", id="point-color", value=Vars.voltage_cfg['point_normal_color'], className="custom-color-picker")
                ], className="d-flex align-items-center justify-content-between mb-2"),


                html.Div([
                    html.Label("테두리 두께", className="m-0"),
                    dbc.Input(id="edge-width", type="number", value=Vars.voltage_cfg['point_edge_width'], size="sm", style={"flex": "0 0 50%"}),
                ], className="d-flex align-items-center justify-content-between mb-3"),
                
                html.Div([
                    html.Label("테두리 색상", className="m-0"),
                    dcc.Input(type="color", id="edge-color", value=Vars.voltage_cfg['point_edge_color'], className="custom-color-picker")
                ], className="d-flex align-items-center justify-content-between mb-2"),

            ], width=4, className="ps-4"),
        ], justify="between", className="mt-5 px-3"),

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
                dbc.Col(html.Div(className="vr mx-2", style={"maxHeight": "100vh", "overflow": "hidden"}), width="auto"),
                dbc.Col(html.Span("NA", className="", 
                    style={
                                "fontSize": "1.2vh",           # 
                                "padding": "0.3em 1.2em",          # 안쪽 여백
                                "borderRadius": "5px",        # 둥근 정도 (Qt의 border-radius)
                                "backgroundColor": "#54365f",  # Qt 스타일의 배경색 (회색 계열)
                                "color": "#ffffff",            # 글자색
                                "fontWeight": "bold",          # 강조
                                "display": "inline-block",
                                "verticalAlign": "middle",     # 수직 정렬 맞춤
                                "marginLeft": "10px"           # 제목과의 간격
                            }), width="auto"),
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

    html.Div([
        # [수정] 수행 시각과 컬러바를 한 줄에 배치
        dbc.Row([
            # 왼쪽: 수행 시각 라벨
            dbc.Col([
                html.Span("수행 시각 : ", style={"fontSize": "1.6vh", "color": "#bbb"}),
                html.Span(initial_time, id="time-display", className="time-text"), 
                html.Span("(SCADA)", className="ms-2 opacity-50", 
                          style={"fontSize": "1.4vh", "backgroundColor": "#15191c", 
                                 "minHeight": "50px",   # 최소한 이 정도 높이는 유지 (모바일이나 작은 창 대비)
                                 "height": "6vh",       # 화면 높이의 6%를 차지하도록 설정
                                 "padding": "0 15px", "display": "flex", "alignItems": "center"
                                 
                                 })
            ], width=True, className="d-flex align-items-center"),

            # 오른쪽: 컬러바(범례) 영역



            dbc.Col([
                html.Div(id="color-bar-legend", className="", style={"height": "100%", "width": "100%", "display": "flex", "alignItems": "center","fontSize": "0.8rem",})
            ], width="auto", className="d-flex align-items-center justify-content-end")
        ], className="mb-2 align-items-center", style={"backgroundColor": "#15191c"}),




        # 메인 캔버스
        html.Div(id="main-graph-canvas", className="canvas-border", style={"height": "75vh",  "backgroundColor": "#15191c"}),
        # 하단 서브 캔버스
        html.Div(id="sub-graph-canvas", className="canvas-border mt-2", style={"height": "10vh",  "width":'50vw', "backgroundColor": "#15191c"})
        
    ], className="main-content m-2", style={"backgroundColor": "#15191c"}),

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
            "display": "flex", # 초기 상태
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



@app.callback(
    [
        Output("main-graph-canvas", "children"), Output("sub-graph-canvas", "children"), Output("color-bar-legend", "children"),
        Output("time-display", "children"), Output("global-loading-overlay", "style")
    ],
    Input("timer-checker", "n_intervals"),
    [State("auto-update-check", "value"), State("time-display", "children"), State("config-store", "data"), State("global-loading-overlay", "style")],
    prevent_initial_call=True
)
def monitor_system(n, auto_update_val, current_display_time, config, current_style):
    global THREAD_STATE
    print(f'[THREAD] : {THREAD_STATE["status"]}')

    #check_thread_context('monitor system')

    # 1. 현재 로딩 중이라면 로딩바 유지하며 아무것도 안 함
    if THREAD_STATE["status"] == "running":
        return [no_update] * 4 + [Vars.loading_on]

    if THREAD_STATE["status"] == "idle":
        return [no_update] * 4 + [{"display": "none"}]


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
                    dcc.Graph(figure=fig_colorbar, config={'displayModeBar': False}, style={"height": "100%", "width": "100%"}), # color-bar-legend
                    data_time, Vars.loading_on 
                ]


    # [Case 2] 자동 갱신이 켜져 있고, 실행 중이 아닐 때 DB 체크
    if auto_update_val and not THREAD_STATE["is_running"]:
        latest_time = get_latest_time_from_db()
        
        # 시간이 달라졌다면 (DB 업데이트 발생)
        if latest_time and str(latest_time) != str(current_display_time):
            run_update_process(config) # 스레드 시작
            return [no_update, no_update,no_update, no_update, Vars.loading_on] # 로딩바 표시


    return [no_update] * 5



@app.callback(
    [
        Output("settings-modal", "is_open"),
        # --- 모달 열 때 UI를 현재 Vars 값으로 초기화 (동기화) ---
        Output("source-radio", "value"), Output("rep-flag-select", "value"), Output("nominal-flag-select", "value"),Output("unit-flag-select", "value"), 
        Output("graph-max-val", "value"), Output("graph-norm-val", "value"), Output("graph-min-val", "value"), Output("grid-resol-input", "value"),
        Output("color-ov", "value"), Output("color-g-max", "value"), Output("color-g-norm", "value"), Output("color-g-min", "value"), Output("color-uv", "value"),
        Output("point-size", "value"), Output("point-color", "value"), Output("edge-width", "value"),Output("edge-color", "value")
    ],
    [
        Input("open-settings", "n_clicks"), Input("btn-accept", "n_clicks"), Input("btn-cancel", "n_clicks")
    ],
    [
        State("settings-modal", "is_open"),
        # 현재 UI에 입력되어 있는 값들 (State)
        State("source-radio", "value"), State("rep-flag-select", "value"), State("nominal-flag-select", "value"), State("unit-flag-select", "value"),
        State("graph-max-val", "value"), State("graph-norm-val", "value"), State("graph-min-val", "value"), State("grid-resol-input", "value"),
        State("color-ov", "value"), State("color-g-max", "value"), State("color-g-norm", "value"), State("color-g-min", "value"), State("color-uv", "value"),
        State("point-size", "value"), State("point-color", "value"), State("edge-width", "value"), State("edge-color", "value")
    ],
    prevent_initial_call=True
)
def handle_modal_logic(n_open, n_accept, n_cancel, is_open, ui_src, ui_rep, ui_nominal, ui_unit, g_max, g_norm, g_min, g_resol,
                       c_ov, c_max, c_norm, c_min, c_uv, p_size, p_color, p_edge_w, p_edge_c):
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return [is_open] + [no_update] * 17

    # 어떤 버튼이 눌렸는지 확인
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # [1] 설정창을 열 때: 전역 변수(Vars) 값을 UI에 "동기화"해서 보여줌
    if triggered_id == "open-settings":
        v = Vars.voltage_cfg
        return [
            True, # 모달 열기, 
            Vars.current_source, Vars.rep_flag, Vars.nominal_flag, Vars.unit_flag,
            v['graph_max_value'], v['graph_normal_value'], v['graph_min_value'], v['level_density'],
            v['graph_ov_color'], v['graph_max_color'], v['graph_normal_color'], v['graph_min_color'], v['graph_uv_color'],
            v['point_size'], v['point_normal_color'], v['point_edge_width'], v['point_edge_color']
        ]

    # [2] 적용(Accept) 버튼을 눌렀을 때만: UI 값을 전역 변수(Vars)에 저장
    if triggered_id == "btn-accept":
        Vars.current_source = ui_src
        Vars.source = ui_src
        Vars.rep_flag = ui_rep
        Vars.nominal_flag = ui_nominal
        Vars.unit_flag = ui_unit



        Vars.voltage_cfg.update({
            "graph_max_value": float(g_max),    
            "graph_normal_value": float(g_norm), 
            "graph_min_value": float(g_min),    
            "level_density": int(g_resol),
            "graph_ov_color": c_ov, "graph_max_color": c_max,
            "graph_normal_color": c_norm, "graph_min_color": c_min, "graph_uv_color": c_uv,
            "point_size": float(p_size), "point_normal_color": p_color,
            "point_edge_width": float(p_edge_w), "point_edge_color": p_edge_c
        })
        # 연산 재시작
        run_update_process(None)

        return [False] + [no_update] * 17

    # [3] 취소(btn-cancel) 혹은 배경 클릭으로 인한 종료: 아무것도 안 하고 닫기만 함
    # 전역 변수를 업데이트하는 로직이 없으므로 '취소'와 동일하게 동작함
    return [False] + [no_update] * 17



#@app.callback(
#    [Output("unit-flag-select", "options"), Output("unit-flag-select", "value")], 
#    [Input("nominal-flag-select", "value")],[State("unit-flag-select", "value")], prevent_initial_call=True 
#)
#def update_unit_options(nominal_val, current_unit):
#    # kV 선택이 불가능한 조건 (ALL 이거나 765+345 조합일 때)
#    is_disabled = nominal_val in ["ALL", "765kV+345kV"]
#    
#    # 1. 옵션 설정 (disabled 속성 부여)
#    new_options = [
#        {"label": "PU", "value": "PU"},
#        {"label": "kV", "value": "kV", "disabled": is_disabled},
#    ]
#    
#    # 2. 값 설정
#    # 비활성화 조건인데 현재 값이 'kV'라면 강제로 'PU'로 변경
#    new_value = no_update
#    if is_disabled and current_unit == "kV":
#        new_value = "PU"
#    
#    return new_options, new_value



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
    run_update_process(None) 
    
    # 브라우저 자동 열기
    Timer(1, open_browser).start()
    
    # 대시 서버 실행
    app.run_server(host="127.0.0.1", port=8050, debug=False) # 디버그 모드 시 스레드 2배 생성 방지




