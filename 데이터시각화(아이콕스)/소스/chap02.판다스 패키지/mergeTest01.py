import pandas as pd

# 7.1.1 데이터 베이스 스타일로 DataFrame 합치기
# DataFrame 2개에 공통적으로 사용되는 컬럼은 key이다.
# 양쪽에 동일한 컬럼이 존재한다.
dict1 = {'name':['홍길동', '홍길동', '김철수', '박영희', '김철수', '김철수', '홍길동'], 'korean': range(7)}
df1 = pd.DataFrame(dict1) 

dict2 = {'name':['김철수', '홍길동', '심수봉'], 'english':range(3)}
df2 = pd.DataFrame(dict2)

print('\n# DataFrame 출력 01')
print(df1)

print('\n# DataFrame 출력 02')
print(df2)

# 출력3 : merge 함수는 하나 이상의 키로 DataFrame의 로우를 합친다.
print('\n# merge() 메소드의 on="name"을 이용하여 데이터 합치기')
print(pd.merge(df1, df2, on='name'))

# how 옵션은 조인 방법을 지정한다.
print("\n# how='outer' 라고 명시하면 full outer join이다.")
print(pd.merge(df1, df2, how='outer'))
# print(pd.merge(df1, df2, how='left'))
# print(pd.merge(df1, df2, how='right'))

print('\n# 컬럼 이름이 동일하지 않는 경우')
dict3 = {'leftkey':['홍길동', '홍길동', '김철수', '박영희', '김철수', '김철수', '홍길동'], 'korean': range(7)}
df3 = pd.DataFrame(dict3)

dict4 = {'rightkey':['김철수', '홍길동', '심수봉'], 'english':range(3)}
df4 = pd.DataFrame(dict4)

print('# DataFrame 출력 03')
print(df3)

print('\n# DataFrame 출력 04')
print(df4)

# 만약 두 객체에 공통되는 컬럼이 하나도 없다면 따로 지정해 주면 된다.
# left_on 항목에는 조인 키로 사용할 왼쪽 DataFrame의 컬럼 이름을 명시한다.
print('\n# merge() 메소드의 left_on과 right_on 사용하기')
print(pd.merge(df3, df4, left_on='leftkey', right_on='rightkey'))
print()

# 여러 개의 키를 병합하려면 컬럼 이름이 들어간 리스트를 넘기면 된다.
dict1 = {'key1':['김철수', '김철수', '박영희'], 
         'key2':['one', 'two', 'one'], 
         'leftval':[1, 2, 3]}
left = pd.DataFrame(dict1)
print('\n# DataFrame 출력 05')
print(left)

dict2 = {'key1':['김철수', '김철수', '박영희', '박영희'], 
         'key2':['one', 'one', 'one', 'two'], 
         'leftval':[4, 5, 6, 7]}
right = pd.DataFrame(dict2)
print('\n# DataFrame 출력 06')
print(right)

# 여러 개의 컬럼을 병합하려면,
# 컬럼 이름이 들어간 리스트를 넘겨 주면 된다.
mylist = ['key1', 'key2'] # 조인할 컬럼 리스트
print('\n# 여러 개의 컬럼 병합하기')
print(pd.merge(left, right, on=mylist, how='outer'))

# 컬럼 2개의 이름이 겹치는 경우에는
# suffixes를 이용하면 이름이 겹치는 컬럼에 대하여 접미사를 지정할 수 있다.
# 특별히 명시하지 않으면 suffixes=('_x', '_y')의 값으로 설정된다.
print('\n# suffixes 옵션 사용하기')
print('# 동일한 컬럼 이름 key2에 대하여 접미사를 붙여 준다.')
print(pd.merge(left, right, on='key1', suffixes=('_왼쪽', '_오른쪽')))

print('\n# 색인을 이용한 머지 사용하기')
newdf1 = df1.set_index('name')
print(newdf1)
print('-' * 40)

newdf2 = df2.set_index('name')
print(newdf2)
print('-' * 40)

print(pd.merge(newdf1, newdf2, left_index=True, right_index=True, how='outer', indicator = True))
print('-' * 40)

print('\n# finished')