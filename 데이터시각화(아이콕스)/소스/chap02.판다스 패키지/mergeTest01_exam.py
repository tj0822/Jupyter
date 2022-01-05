import pandas as pd

dict1 = {'이름':['유관순', '유관순', '황진이', '황진이', '유관순'], '중간': [10*idx for idx in range(1, 6)]}
df1 = pd.DataFrame(dict1)
print(df1)
print('-'*30)

dict2 = {'이름':['황진이', '유관순', '신사임당'], '기말':[10*idx for idx in range(3, 6)]}
df2 = pd.DataFrame(dict2)
print(df2)
print('-'*30)

print(pd.merge(df1, df2, on='이름'))
print('-'*30)

print(pd.merge(df1, df2, how='outer'))
print('-'*30)