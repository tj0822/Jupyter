import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')
cnt, PNG, UNDERBAR = 0, '.png', '_'
CHART_NAME = 'SpecialPlot'
filename = './../data/welfareClean.csv'
plt.rcParams['axes.unicode_minus'] = False

welfare = pd.read_csv(filename, encoding='cp949')
print(welfare.columns)
print(welfare.head())
################################################################################
print('범주형 플로팅')
################################################################################
fig = plt.figure(figsize=(16, 10), dpi= 80)

sns.catplot('성별', col='결혼 유무', col_wrap=3,
                data=welfare, kind="count", height=3.5,
                aspect=.8, palette='tab20')

fig.suptitle('제목')

cnt += 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')
################################################################################
print('Density Plotting')
################################################################################
print("welfare['결혼 유무'].unique()")
print(welfare['결혼 유무'].unique())

plt.figure(figsize=(16,10), dpi= 80)
MY_ALPHA = 0.6
sns.kdeplot(welfare.loc[welfare['결혼 유무'] == '무응답', '나이'], shade=True, color="r", label="무응답", alpha=MY_ALPHA)
sns.kdeplot(welfare.loc[welfare['결혼 유무'] == '이혼', '나이'], shade=True, color="g", label="이혼", alpha=MY_ALPHA)
sns.kdeplot(welfare.loc[welfare['결혼 유무'] == '결혼', '나이'], shade=True, color="b", label="결혼", alpha=MY_ALPHA)

plt.title('결혼 유무에 따른 나이의 밀도 곡선', fontsize=22)
plt.legend()

cnt += 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')
################################################################################
'''
The lollipop chart is a composite chart with bars and circles.
It is a variant of the bar chart with a circle at the end, to highlight the data value.
Like a bar chart, a lollipop chart is used to compare categorical data
'''
print('Lollipop Chart')
################################################################################
df = welfare[['소득', '지역구']].groupby('지역구').apply(lambda x: x.mean())
df.sort_values('소득', inplace=True)
df.reset_index(inplace=True)

print('df')
print(df)

# Draw plot
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)

# 수직선
ax.vlines(x=df.index, ymin=0, ymax=df.소득, color='firebrick', alpha=0.7, linewidth=2)

# 상단의 점
ax.scatter(x=df.index, y=df.소득, s=75, color='firebrick', alpha=0.7)

# Title, Label, Ticks and Ylim
ax.set_title('지역구를 위한 Lollipop Chart', fontdict={'size':22})
ax.set_ylabel('소득의 평균')
ax.set_xticks(df.index)
ax.set_xticklabels(df.지역구.str.lower(), rotation=30, fontdict={'horizontalalignment': 'right', 'size':12})

# 상단에 그려 주는 소득 정보 텍스트
for row in df.itertuples():
    ax.text(row.Index, row.소득+.5, s=round(row.소득, 2), horizontalalignment= 'center', verticalalignment='bottom', fontsize=14)

cnt += 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')
################################################################################
print('Ordered Bar Chart')
################################################################################
df = welfare[['소득', '지역구']].groupby('지역구').apply(lambda x: x.mean())
df.sort_values('소득', inplace=True) # gallon 당 고속도로 주행 마일 수
df.reset_index(inplace=True)

print('df')
print(df)

print('df.index')
print(df.index)

# Draw plot
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(16,10), facecolor='white', dpi= 80)
ax.vlines(x=df.index, ymin=0, ymax=df.소득, color='firebrick', alpha=0.7, linewidth=20)

# Annotate Text : Bar의 상단에 수치 데이터를 보여 줍니다.
for i, income in enumerate(df.소득):
    ax.text(i, income+0.5, round(income, 1), horizontalalignment='center')

# 그래프 제목, y축 상하한선, c축 Ticks 정의
ax.set_title('Ordered Bar Chart', fontdict={'size':22})
ax.set(ylabel='지역구별 소득의 평균', ylim=(0, 301))
plt.xticks(df.index, df.지역구.str.lower(), rotation=60, horizontalalignment='right', fontsize=12)

# # Add patches to color the X axis labels
# p1 = patches.Rectangle((.57, -0.005), width=.33, height=.13, alpha=.3, facecolor='red', transform=fig.transFigure)
# p2 = patches.Rectangle((.124, -0.005), width=.446, height=.13, alpha=.3, facecolor='green', transform=fig.transFigure)
# fig.add_artist(p1)
# fig.add_artist(p2)

cnt += 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')
################################################################################
print('TreeMap')
################################################################################
# pip install squarify
import squarify

df = welfare.groupby('지역구').size().reset_index(name='counts')
print('df')
print(df)

labels = df.apply(lambda x: str(x[0]) + "\n (" + str(x[1]) + ")", axis=1)
print('labels') # 각 셀에 들어갈 레이블 텍스트
print(labels)

sizes = df['counts'].values.tolist()

print('sizes')
print(sizes)

colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]

print('colors')
print(colors)

# Draw Plot
plt.figure(figsize=(12,8), dpi= 80)
squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)

# Decorate
plt.title('Treemap 다루기')
plt.axis('off')

cnt += 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')
################################################################################
print('Stem Plotting')
################################################################################
import numpy as np
newwelfare = welfare['소득'].head(100)

print('newwelfare')
print(newwelfare)

plt.figure()
x = np.arange(len(newwelfare))
y = np.array(newwelfare)

plt.stem(x, y)

cnt += 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')
################################################################################