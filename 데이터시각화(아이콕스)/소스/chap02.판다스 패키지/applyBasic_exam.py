import pandas as pd

filename = './../data/memberInfo_exam.csv'
df = pd.read_csv(filename, index_col='이름')
print(df)
print('-'*30)

def sungjuk(x):
    result = ''
    if x >= 90 :
        result = '수'
    elif x >= 80 :
        result = '우'    
    elif x >= 70 :
        result = '미'    
    elif x >= 60 :
        result = '양'    
    else:
        result = '가'            
    return result


sq = df['중간'].apply(sungjuk) 
print(sq)
print('-'*30)