import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')
cnt, PNG, UNDERBAR = 0, '.png', '_'
CHART_NAME = 'slopeChart_exam'
filename = './../data/매출.csv'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(filename, encoding='cp949')

fig, ax = plt.subplots(1, 1, figsize=(14, 14), dpi= 80)

HUNDRED = 90
MIN_HEIGHT = 20
MAX_HEIGHT = HUNDRED + 5

# 수직선 그리기
ax.vlines(x=1, ymin=MIN_HEIGHT, ymax=MAX_HEIGHT, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=MIN_HEIGHT, ymax=MAX_HEIGHT, color='black', alpha=0.7, linewidth=1, linestyles='dotted')

import numpy as np

ax.scatter(y=df['일사분기'], x=np.repeat(1, df.shape[0]), s=10, color='black', alpha=0.7)
ax.scatter(y=df['이사분기'], x=np.repeat(3, df.shape[0]), s=10, color='black', alpha=0.7)


idx = 0

left_label = [str(c) + '(' + str(round(y)) + '점)' for c, y in zip(df.name, df['일사분기'])]
print('left_label')
print(left_label)

right_label = [str(c) + '(' + str(round(y)) + '점)' for c, y in zip(df.name, df['이사분기'])]

import matplotlib.lines as mlines

def newline(p1, p2):
    ax = plt.gca()
    l = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]], color='red' if p1[1]-p2[1] > 0 else 'green', marker='o', markersize=6)
    ax.add_line(l)
    return l

for p1, p2 in zip(df['일사분기'], df['이사분기']):
    newline([1, p1], [3, p2]) # 연결선을 그려 주고
    # 선들에 대한 caption 작성하기
    ax.text(1-0.05, p1, left_label[idx], horizontalalignment='right', verticalalignment='center', fontdict={'size':14})
    ax.text(3+0.05, p2, right_label[idx], horizontalalignment='left', verticalalignment='center', fontdict={'size':14})
    idx = idx + 1

ax.text(1-0.05, HUNDRED + 5, '일사분기', horizontalalignment='right', verticalalignment='center', fontdict={'size':18, 'weight':700})
ax.text(3+0.05, HUNDRED + 5, '이사분기', horizontalalignment='left', verticalalignment='center', fontdict={'size':18, 'weight':700})

# Decoration
ax.set_title('두 분기의 과일 매출', fontdict={'size':22})
ax.set(xlim=(0,4), ylim=(MIN_HEIGHT, MAX_HEIGHT), ylabel='매출 추이')

# 하단의 x축에 대한 ticks 작성
ax.set_xticks([1, 3])
ax.set_xticklabels(['일사분기', '이사분기'])
# plt.yticks(np.arange(500, 13000, 2000), fontsize=12)

# 그래프 영역의 테두리 선(borders) 없애기
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)

cnt += 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')