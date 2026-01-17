metro_area = [('Tokyo','JP', 36.933, (35.689722, 139.691667))]


print(f'{"":15} | {"latitude": >9} | {"longitude":>9}')
for name, _, _, (lat, lon) in metro_area:
    print(f'{name:15} | {lat:9.4f} | {lon:9.4f}')
    


# 3.10 이후 match/case 사용 가능
# C 의 swtich/case 문과 유사
# *(asterisk) 를 사용한 rest parameter unpacking 활용 방법

