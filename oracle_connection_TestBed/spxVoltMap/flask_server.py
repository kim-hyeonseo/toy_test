from flask import Flask, jsonify
import oracledb
import pandas as pd

import oracledb
import requests
import json
# 오라클 DB 연결 정보 설정
host = '164.124.101.2'  # 확인한 호스트 이름
port = "1521"  # 기본 오라클 포트
service_name = "aextpdb.dbtoolsprdphxae.dbtoolsprodprod.oraclevcn.com"  # 확인한 서비스 이름
username = "WKSP_HSTEST"  # 확인한 스키마 이름 (DB 사용자명)
password = "1q2w3e4r!!K"  # 비밀번호 (사용자가 설정한 값)

# DSN (SERVICE_NAME 방식 적용)
dsn = f"{host}:{port}/{service_name}"

print("Generated DSN:", dsn)

# Flask 애플리케이션 생성
app = Flask(__name__)

# 오라클 연결 설정 함수
def get_db_connection():
    try:
        connection = oracledb.connect(
            user=username, 
            password=password, 
            dsn=dsn, 
            mode=oracledb.DEFAULT_AUTH
        )

        print("Connected to Oracle DB successfully!")
        return connection
        
    except oracledb.DatabaseError as e:
        print("Connection failed:", e)
        return None

# 데이터 조회 함수
def fetch_data():
    print("Fetching data via Flask...")
    conn = get_db_connection()
    if not conn:
        return {"error": "Database connection failed"}
    
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM contour_sample ORDER BY ss"
        cursor.execute(query)
        rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=column_names)
    except Exception as e:
        print("Error fetching data:", e)
        df = pd.DataFrame()
    finally:
        cursor.close()
        conn.close()
    return df

# Flask 라우트
@app.route('/get_data', methods=['GET'])
def get_data():
    i=0
    df = fetch_data()
    if "error" in df:
        return jsonify(df), 500
    return jsonify(df.to_dict(orient='records'))
>>>>>>> cd472e29037a7b762a8695a131f3615afb04ce7d

# 오라클 DB 연결 정보 설정
host = '164.124.101.2'  # 확인한 호스트 이름
port = "1521"  # 기본 오라클 포트
service_name = "aextpdb.dbtoolsprdphxae.dbtoolsprodprod.oraclevcn.com"  # 확인한 서비스 이름
username = "WKSP_HSTEST"  # 현재 스키마 사용자 이름
password = "1q2w3e4r!!K"  # 비밀번호 (사용자가 설정한 값)

# DSN (SERVICE_NAME 방식 적용)
dsn = f"{host}:{port}/{service_name}"

print("Generated DSN:", dsn)
