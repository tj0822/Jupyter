import pandas as pd

myindex = ['서울', '부산', '광주', '대구', '울산', '목포', '여수']
mylist = [50, 60, 40, 80, 70, 30, 20]
myseries = pd.Series(data=mylist, index=myindex)
print(myseries)

print('\n색인의 이름으로 값 읽기')
print(myseries[['대구']])

print('\n라벨 이름으로 슬라이싱')
print(myseries['대구':'목포'])

print('\n여러 개의 색인 이름으로 데이터 읽기')
print(myseries[['대구', '여수']])

print('\n정수를 이용한 데이터 읽기')
print(myseries[[2]])

print('\n0, 2, 4번째 데이터 읽기')
print(myseries[0:5:2])

print('\n1, 3, 5번째 데이터 읽기')
print(myseries[[1, 3, 5]])

print('\n슬라이싱 사용하기')
print(myseries[3:6]) # from <= 결과 < to

print('\n2번째 항목의 값 변경')
myseries[2] = 22

print('\n2번째부터 4번째 까지 항목의 값 변경')
myseries[2:5] = 33

print('\n서울과 대구만  55로 변경')
myseries[['서울', '대구']] = 55

print('\n짝수 행만  77로 변경')
myseries[0::2] = 77

print('\n시리즈 내용 확인')
print(myseries)

print('\nfinished')