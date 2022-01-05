import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
###############################################################################
plt.rc('font', family='Malgun Gothic')
cnt, PNG, UNDERBAR = 0, '.png', '_'
CHART_NAME = 'brokenLineExam'
filename = './../data/주요발생국가주간동향(4월2째주).csv'
###############################################################################
data = pd.read_csv(filename, index_col='국가')
print(data.columns)
print('-' * 30)

chartdata = data['4월06일']
print(chartdata)
print('-' * 30)
print(type(chartdata))

YTICKS_INTERVAL = 50000
maxlim = (int(chartdata.max() / YTICKS_INTERVAL) + 1) * YTICKS_INTERVAL

plt.plot(chartdata, color='blue', linestyle='solid', marker='o')

values = np.arange(0, maxlim + 1, YTICKS_INTERVAL)
plt.yticks(values, ['%s' % format(val, ',') for val in values])

plt.grid(True)
plt.xlabel('국가명')

plt.ylabel("발생 건수")
plt.title('4월 6일 코로나 발생 건수')

cnt += 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')

COUNTRY = ['스페인', '프랑스', '독일', '중국', '영국', '이란']
WHEN = ['4월06일', '4월07일', '4월08일', '4월09일', '4월10일']
chartdata = data.loc[COUNTRY, WHEN]

chartdata = chartdata.T

print(chartdata.columns)
print('-' * 30)

print(chartdata)
print('-' * 30)

chartdata.plot(title='SomeTitie', figsize=(10, 6), legend = True, marker='o', rot=0)

plt.grid(True)
plt.xlabel('일자')
plt.ylabel('국가명')
plt.title('일자별 국가명 꺽은 선')

cnt += 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')

tipsfile = './../data/tips.csv'

myframe = pd.read_csv(tipsfile)

myframe = myframe.head(100)

print(myframe.info())
print('-' * 30)

print(myframe.describe())
print('-' * 30)

print(len(myframe))
print('-' * 30)

xrange = range(len(myframe))
data_bill = myframe.loc[:, ['total_bill']]
data_tip = myframe.loc[:, ['tip']]

fig, ax1 = plt.subplots()

ax1.set_title('결재 금액과 Tip(이중 축)')

color = 'tab:red'

ax1.set_ylabel('결재 금액', color=color)
ax1.plot(xrange, data_bill, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('팁(tip)', color=color)  # we already handled the x-label with ax1
ax2.plot(xrange, data_tip, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# otherwise the right y-label is slightly clipped
fig.tight_layout()

cnt = cnt + 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')
###############################################################################
print('finished')