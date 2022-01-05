import pandas as pd

myindex = ['마포구', '용산구', '서대문구', '동대문구', '은평구', '구로구', '강서구']
mylist = [40, 80, 70, 50, 60, 30, 20]
myseries = pd.Series(data=mylist, index=myindex)
print(myseries)
print('-'*30)

# 색인의 이름으로 값 읽기
print(myseries[['은평구']])
print('-'*30)

# 라벨 이름으로 슬라이싱
print(myseries['서대문구':'구로구'])
print('-'*30)

# 여러 개의 색인 이름으로 데이터 읽기
print(myseries[['용산구', '동대문구']])
print('-'*30)

# 정수를 이용한 데이터 읽기
print(myseries[[2]])
print('-'*30)

# 0, 2, 4번째 데이터 읽기
print(myseries[0:5:2])
print('-'*30)

# 1, 3, 4번째 데이터 읽기
print(myseries[[1, 3, 4]])
print('-'*30)

# 슬라이싱 사용하기
print(myseries[2:5]) # from <= 결과 < to

myseries[2] = 99 # 2번째 항목의 값 변경
myseries[2:4] = 66 # 2번째부터 4번째 까지 항목의 값 변경
myseries[['마포구', '강서구']] = 55 # '마포구'와 '강서구'만  55로 변경
myseries[0::2] = 77 # 짝수 행만  77로 변경

# 시리즈 내용 확인
print(myseries)
print('-'*30)