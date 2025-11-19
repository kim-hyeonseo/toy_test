# app.py
import os
from datetime import datetime
from flask import Flask, render_template, request, send_file, url_for, send_file, session, redirect
from dotenv import load_dotenv
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping

# 한글 폰트 등록 (한 번만 하면 됨)
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows 기준 (맑은 고딕)
pdfmetrics.registerFont(TTFont("Malgun", font_path))
addMapping("Malgun", 0, 0, "Malgun")



from logic import run_my_program



load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret')
PORT = int(os.getenv('PORT', 5000))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ✅ 이 부분이 반드시 함수들보다 위에 있어야 함
questions = [
    {"id": f"Q{i}", "text": f"문항 {i}번 질문입니다."}
    for i in range(1, 11)
]



@app.route('/')
def home():
    session.clear()  # 시작 시 세션 초기화
    return redirect(url_for('question', num=1))

@app.route('/question/<int:num>', methods=['GET', 'POST'])
def question(num):
    if request.method == 'POST':
        choice = request.form.get('choice')
        if choice:
            # 세션에 답변 저장
            answers = session.get('answers', {})
            answers[f"Q{num}"] = int(choice)
            session['answers'] = answers

            # 다음 문항으로 이동
            if num < len(questions):
                return redirect(url_for('question', num=num + 1))
            else:
                return redirect(url_for('result'))

    q = questions[num - 1]
    return render_template('question.html', q=q, num=num, total=len(questions))

@app.route('/result')
def result():
    answers = session.get('answers', {})
    total_score = sum(answers.values())

    # PDF 생성
    pdf_path = "result.pdf"
    c = canvas.Canvas(pdf_path, pagesize=A4)
    c.setFont("Malgun", 14)
    c.drawString(72, 800, "설문 결과 요약")
    c.setFont("Malgun", 12)
    y = 770
    for k, v in answers.items():
        c.drawString(72, y, f"{k}: {v}점")
        y -= 25
    c.drawString(72, y - 10, f"총점: {total_score}점")
    c.save()

    return render_template('result.html', answers=answers, total=total_score, pdf_file='result.pdf')

@app.route('/download/<path:filename>')
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)