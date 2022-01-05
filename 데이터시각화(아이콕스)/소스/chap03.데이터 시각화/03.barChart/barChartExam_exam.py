import numpy as np
import matplotlib.pyplot as plt
###############################################################################
plt.rc('font', family='Malgun Gothic')
cnt, PNG, UNDERBAR = 0, '.png', '_'
CHART_NAME = 'barChartExam_연습_문제'
filename = './../data/final_exam.csv'
###############################################################################
import pandas as pd

data = pd.read_csv(filename, index_col='names', encoding='cp949')

print(data.columns)
print('-'*30)

chartdata = data['korean']
print(chartdata)
print('-'*30)
print('type(chartdata)')
print(type(chartdata)) # Series
###############################################################################
# plt.bar() 메소드를 사용한 막대 그래프
def MakeBarChart01(x, y, color, xlabel, ylabel, title):
    plt.figure()
    plt.bar(x, y, color=color, alpha=0.7)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.grid(True)

    YTICKS_INTERVAL = 10

    maxlim = (int(y.max() / YTICKS_INTERVAL) + 1) * YTICKS_INTERVAL
    print(maxlim)

    values = np.arange(0, maxlim + 1, YTICKS_INTERVAL)

    plt.yticks(values, ['%s' % format(val, ',') for val in values])

    # 그래프 위에 건수와 비율 구하기
    ratio = 100 * y / y.sum()
    print(ratio)
    print('-' * 40)

    plt.rc('font', size=6)
    for idx in range(y.size):
        value = format(y[idx], ',') + '점'
        plt.text(x=idx, y=y[idx] + 1, s=value, horizontalalignment='center')

    # 평균 값을 수평선으로 그리기
    meanval = y.mean()
    print(meanval)
    print('-' * 40)

    average = '평균 : %d점' % meanval
    plt.axhline(y=meanval, color='r', linewidth=1, linestyle='dashed')
    plt.text(x=0, y=meanval + 1, s=average, horizontalalignment='center')

    global cnt
    cnt = cnt + 1
    savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
    plt.savefig(savefile, dpi=400)
    print(savefile + ' 파일이 저장되었습니다.')
# def MakeBarChart01
###############################################################################
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
mycolor = colors[0:len(chartdata)]

MakeBarChart01(x=chartdata.index, y=chartdata, color=mycolor, xlabel='학생 이름', ylabel='점수', title='국어 점수')
