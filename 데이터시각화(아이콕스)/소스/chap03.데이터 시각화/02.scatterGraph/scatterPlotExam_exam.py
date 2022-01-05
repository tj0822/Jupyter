import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')
cnt, PNG, UNDERBAR = 0, '.png', '_'
CHART_NAME = 'scatterPlotExam_연습_문제'
filename = './../data/kbo.csv'
plt.style.use('ggplot')

kbo = pd.read_csv(filename, encoding='cp949')

xdata = kbo.loc[:, ['AVG']]
ydata = kbo.loc[:, ['HR']]

plt.figure()
plt.plot(xdata, ydata, marker='o', linestyle='None' )
plt.xlabel("타율")
plt.ylabel("홈런")
plt.title("산점도 그래프")
plt.grid(True)

cnt += 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')