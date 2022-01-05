import pandas
df = pandas.read_csv('./../data/welfare.csv')

print(df.head())

print(type(df))

print(df.shape)
print(df.shape[0])
print(df.shape[1])

print(df.columns)

print(df.dtypes)

print(df.info())

gender_df = df['gender']

print(type(gender_df))

print(gender_df.head())

print(gender_df.tail())

subset = df[['gender', 'birth', 'marriage']]

print(type(subset))

print(subset.head())

print(subset.tail())

# loc 속성으로 행 단위 데이터 추출

print(df.loc[0])

print(df.loc[99])

number_of_rows = df.shape[0]
last_row_index = number_of_rows - 1

print(df.loc[last_row_index])

print(df.tail(n=1))

print(df.tail(n=2))

print(df.loc[[0, 99, 999]])

# tail 메서드와 loc 속성이 반환하는 자료형은 서로 다릅니다.
subset_loc = df.loc[0]
subset_tail = df.tail(n=1)

print(type(subset_loc))
print(type(subset_tail))

# iloc 속성으로 행 데이터 추출
print(df.iloc[1])

print(df.iloc[99])

print(df.iloc[-1])

print(df.iloc[1710])

print(df.iloc[[0, 99, 999]])

# 슬라이싱 구문으로 데이터 추출
subset = df.loc[:, ['marriage', 'income']]
print(subset.head())

subset = df.iloc[:, [2, 4, -1]]
print(subset.head())

#range 메서드를 사용한 데이터 추출
small_range = list(range(5))
print(small_range)

print(type(small_range))

subset = df.iloc[:, small_range]
print(subset.head())

small_range = list(range(3, 6))
print(small_range)

subset = df.iloc[:, small_range]
print(subset.head())

small_range = list(range(0, 6, 2))
subset = df.iloc[:, small_range]
print(subset.head())

# 그룹화한 데이터의 평균 구하기

print(df.head(n=10))

print(df['marriage'].unique())

print(df.groupby('marriage')['code_religion'].mean())

mygrouping = df.groupby('marriage')
print(type(mygrouping))

print(mygrouping)

grp_code_religion = mygrouping['code_religion']
print(type(grp_code_religion))

mean_code_religion = grp_code_religion.mean()
print(mean_code_religion)

multi_group_var = df.groupby(['marriage', 'birth'])[['code_religion', 'income']].mean()
print(multi_group_var)

print(type(multi_group_var))