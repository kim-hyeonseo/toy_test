# logic.py
# 여기에 당신의 프로그램 로직을 함수로 감싸두고, Flask에서 호출합니다.

def run_my_program(user_text: str) -> str:
    """
    예시: 입력 텍스트를 처리해서 결과 문자열을 돌려준다.
    실제로는 exe 호출, AI 모델 처리, CSV 분석 등으로 바꿔도 됨.
    """
    # TODO: 여기서 .exe 호출하거나, 내부 라이브러리/모델 로직을 사용
    # 예시로 간단히 역순 반환
    return user_text[::-1]

