"""
사용자 정의 색상을 사용한 상자 수염 그래프 그리기
두 가지 유형(rectangular and notched)의 상자 수염 그래프를 그려 봅니다.
부가적으로 `labels` 매개 변수를 이용하여 x-tick labels을 지정해 봅니다.

참조 파일) http://vita.had.co.nz/papers/boxplots.pdf
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
###############################################################################
plt.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False
cnt, PNG, UNDERBAR = 0, '.png', '_'
CHART_NAME = 'boxPlotExam'
filename = './../data/tips.csv'
###############################################################################
myframe = pd.read_csv(filename, encoding='utf-8', index_col=0)

DINNER, LUNCH = 'Dinner', 'Lunch'

frame01 = myframe.loc[myframe['time'] == DINNER, 'total_bill']
frame01.index.name = DINNER

frame02 = myframe.loc[myframe['time'] == LUNCH, 'total_bill']
frame02.index.name = LUNCH

chartdata = [np.array(frame01), np.array(frame02)]
print('chartdata')
print(chartdata)

xtick_label = [DINNER, LUNCH] # x 축을 위한 레이블

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

# rectangular box plot
bplot1 = ax1.boxplot(chartdata,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=xtick_label)  # will be used to label x-ticks
ax1.set_title('Rectangular box plot')

# notch shape box plot
bplot2 = ax2.boxplot(chartdata,
                     notch=True,  # notch shape
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=xtick_label)  # will be used to label x-ticks
ax2.set_title('Notched box plot')

colors = ['pink', 'lightblue'] # fill with colors
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

# adding horizontal grid lines
for ax in [ax1, ax2]:
    ax.yaxis.grid(True)
    ax.set_xlabel('점심과 저녁 팁')
    ax.set_ylabel('observed data')

cnt += 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')

print('finished')