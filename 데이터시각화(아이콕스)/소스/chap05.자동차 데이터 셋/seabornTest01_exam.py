import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
plt.rcParams["font.size"] = 12
plt.rcParams['xtick.labelsize'] = 12.
plt.rcParams['ytick.labelsize'] = 12.
plt.rcParams['axes.unicode_minus'] = False

CHART_NAME = 'seabornTest_exam'
cnt, PNG, UNDERBAR = 0, '.png', '_'
filename = './../data/tips.csv'
# filename = './../data/tips.csv'

tips = pd.read_csv(filename, encoding='utf-8', index_col=0)

print(tips.columns)
print('-'*30)

print(tips.head(10))
print('-'*30)

print(tips.describe())
print('-'*30)

def FileSave():
    global cnt
    cnt += 1
    savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
    plt.savefig(savefile, dpi=400)
    print(savefile + ' 파일이 저장되었습니다.')
# end def FileSave():

import seaborn as sns

total_bill = tips['total_bill'] # Series
print('total_bill.unique()')
print(total_bill.unique())

import numpy as np
print(np.max(total_bill.unique()))
print(np.min(total_bill.unique()))

ax = plt.subplots()
sns.scatterplot(data=tips, x='total_bill', y='tip')
ax = sns.rugplot(data=tips, x='total_bill', y='tip')
ax.set_title('산점도와 Rug Plot')
ax.set_xlabel('총 지급액')
ax.set_ylabel('팁(tip)')
FileSave()

ax = plt.subplots()
ax = sns.histplot(data=tips, x='total_bill', kde=True, bins=30)
ax.set_title('총 지급액 히스토그램')
ax.set_xlabel('총 지급액')
ax.set_ylabel('')
FileSave()

print("tips['day'].unique()")
print(tips['day'].unique())

label_dict = {'Sun':'일요일', 'Sat':'토요일', 'Thur':'목요일', 'Fri':'금요일'}

def setLabel(x):
    return label_dict[x]

idx = 0 # 색상 구분을 위한 카운터 변수
tips['day'] = tips['day'].apply(setLabel)

print('요일별 갯수')
result = tips.groupby('day')['day'].count()
print(result)

print("tips['time'].unique()")
print(tips['time'].unique())

print('주간/야간 구분 개수')
result = tips.groupby('time')['time'].count()
print(result)

plt.figure()
ax = sns.barplot(x='day', y='tip', hue='time', data=tips, errwidth=0)
ax.set_title("요일에 따른 팁(tip))")
ax.set_xlabel('요일')
ax.set_ylabel('팁(tip)')
FileSave()

print('요일에 따른 따른 팁(tip)의 통계치 정보')
mygrouping = tips.groupby('day')['tip']
print(mygrouping.describe())

plt.figure()
ax = sns.boxplot(x='day', y='tip', hue='time', data=tips)
ax.set_title('요일에 따른 따른 팁(tip)(boxplot)')
ax.set_xlabel('요일')
ax.set_ylabel('팁(tip)')
FileSave()

plt.figure()
ax = sns.violinplot(x='day', y='tip', hue='time', data=tips)
ax.set_title('요일에 따른 팁(tip)(violinplot)')
ax.set_xlabel('요일')
ax.set_ylabel('팁(tip)')
FileSave()

# UserWarning: 5.7% of the points cannot be placed;
# you may want to decrease the size of the markers or use stripplot.
STRIP_SIZE = 3

plt.figure()
ax = sns.stripplot(x='day', y='tip', hue='time', data=tips, jitter=True, size=STRIP_SIZE)
ax.set_title('요일에 따른 팁(tip)(stripplot)')
ax.set_xlabel('요일')
ax.set_ylabel('팁(tip)')
plt.legend(loc=1)
FileSave()


plt.figure()
ax = sns.swarmplot(x='day', y='tip', hue='time', data=tips)
ax.set_title("Swarm Plot(hue='time')")
ax.set_xlabel('요일')
ax.set_ylabel('팁(tip)')
plt.legend(loc=1)
FileSave()


import numpy as np
plt.figure()
sns.boxplot(x='total_bill', y='day', data=tips, whis=np.inf)
ax = sns.stripplot(x='total_bill', y='day', data=tips, jitter=True, color="0.4")
ax.set_title("Boxplot과 Strip Plot로 표현")
ax.set_xlabel('총 지급액')
ax.set_ylabel('요일')
FileSave()