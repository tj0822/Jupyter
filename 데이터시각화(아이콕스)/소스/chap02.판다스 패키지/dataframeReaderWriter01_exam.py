import numpy as np 
import pandas as pd

myindex = ['김구', '이봉창', '안중근', '윤봉길']
mycolumns = ['강남구', '은평구', '마포구', '용산구']
mylist = list(10 * onedata for onedata in range(1, 17))
myframe = pd.DataFrame(np.reshape(mylist, (4, 4)), index=myindex, columns = mycolumns)

print(myframe)
print('-'*30)

result = myframe.iloc[1]
print(result)
print('-'*30)

result = myframe.iloc[[1,3]]
print(result)
print('-'*30)

result = myframe.loc[['윤봉길']]
print(result)
print('-'*30)

result = myframe.loc[['이봉창', '윤봉길']]
print(result)
print('-'*30)

result = myframe.loc[['윤봉길'], ['은평구']] # DataFrame
print(result)
print('-'*30)

result = myframe.loc[['김구', '이봉창'], ['용산구', '은평구']]
print(result)
print('-'*30)

result = myframe.loc[ myframe['은평구'] <= 100 ]
print(result)
print('-'*30)

result = myframe.loc[ myframe['은평구'] == 100 ]
print(result)
print('-'*30)


myframe.loc['김구':'안중근', ['용산구']] = 80
print(myframe)
print('-'*30)