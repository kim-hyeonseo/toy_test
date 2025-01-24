from flask import Flask, jsonify
import psycopg2
import psycopg2.extras
import pandas as pd


import requests
print('start flag')
rest_api_url = "https://apex.oracle.com/pls/apex/YOUR_WORKSPACE/hr/employees"
print('flag1')


try:
    response = requests.get(rest_api_url, timeout=5)  # 10초 내 응답 없으면 중단
    print('flag2')

    if response.status_code == 200:
        print("API Data:", response.json())
    else:
        print("Failed to fetch data, Status:", response.status_code)

except requests.exceptions.Timeout:
    print("Error: The request timed out")
except requests.exceptions.ConnectionError:
    print("Error: Unable to connect to the server")
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")




if response.status_code == 200:
    print("API Data:", response.json())
else:
    print("Failed to fetch data, Status:", response.status_code)



#import oracledb
#import requests
#import json

## 오라클 DB 연결 정보 설정 (APEX에서 확인한 값 적용)
#host = '164.124.101.2'  # 확인한 호스트 이름
#port = "1521"  # 기본 오라클 포트
#service_name = "aextpdb.dbtoolsprdphxae.dbtoolsprodprod.oraclevcn.com"  # 확인한 서비스 이름
#username = "WKSP_HSTEST"  # 확인한 스키마 이름 (DB 사용자명)
#password = "1q2w3e4r!!K"  # 비밀번호 (사용자가 설정한 값)

## DSN (SERVICE_NAME 방식 적용)
#dsn = f"{host}:{port}/{service_name}"

#print("Generated DSN:", dsn)

## RESTful API 엔드포인트 (APEX에서 설정한 URL)
#rest_api_url = "https://apex.oracle.com/pls/apex/hs_test/hr/employees"

#try:
#    # 오라클 데이터베이스 연결 테스트
#    connection = oracledb.connect(user=username, password=password, dsn=dsn)
#    print("Connected to Oracle DB successfully!")

#    # REST API 호출로 샘플 데이터 가져오기
#    response = requests.get(rest_api_url)

#    if response.status_code == 200:
#        data = response.json()
#        print("REST API Data:")
#        for record in data['items']:
#            print(record)
#    else:
#        print(f"Failed to fetch data, HTTP Status: {response.status_code}")

#    connection.close()
#except oracledb.DatabaseError as e:
#    print("Connection failed:", e)
#except Exception as ex:
#    print("Unexpected error:", ex)

## Thin 모드 확인q
#try:
#    print("Is Thin Mode:", connection.thin)
#except NameError:
#    print("Connection was not established, skipping Thin mode check.")



## Flask 애플리케이션 생성
#app = Flask(__name__)

## PostgreSQL 연결 설정 함수
#def get_db_connection():

#    return psycopg2.connect(
#        host="localhost",        # PostgreSQL 서버 호스트
#        dbname="postgres",       # 데이터베이스 이름
#        user="postgres",         # 사용자 이름
#        password="5432",# 비밀번호
#        port=5432                # 포트 번호https://desktop.postman.com/_ar-assets/images/quickstart-banner-dark-0d0e3ff497282c09401613aacc082f48.svg
#    )

## 데이터 조회 함수
#def fetch_data():
#    print("Fetching data via Flask...")  # Flask를 통해 데이터를 가져옴을 로그로 출력

#    conn = get_db_connection()
#    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

#    query = "SELECT * FROM public.contour_sample ORDER BY ss"
#    cur.execute(query)
#    rows = cur.fetchall()
#    column_names = [desc[0] for desc in cur.description]

#    # 데이터를 DataFrame으로 변환
#    df = pd.DataFrame(rows, columns=column_names)
#    cur.close()
#    conn.close()
#    return df

## Flask 라우트
#@app.route('/get_data', methods=['GET'])
#def get_data():
#    df = fetch_data()
#    return jsonify(df.to_dict(orient='records'))

#if __name__ == '__main__':
#    app.run(debug=True, port=5000)
