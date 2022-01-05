import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
plt.rcParams["font.size"] = 12
plt.rcParams['xtick.labelsize'] = 12.
plt.rcParams['ytick.labelsize'] = 12.
plt.rcParams['axes.unicode_minus'] = False

CHART_NAME = 'seabornTest'
cnt, PNG, UNDERBAR = 0, '.png', '_'
filename = './../data/mpg.csv'

mpg = pd.read_csv(filename, encoding='utf-8')

print(mpg.columns)
# ['manufacturer', 'model', 'displ', 'year', 'cyl', 'trans', 'drv', 'cty', 'hwy', 'fl', 'class']
print('-'*30)

print(mpg.head(10))
print('-'*30)

print(mpg.describe())
print('-'*30)

def FileSave():
    global cnt
    cnt += 1
    savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
    plt.savefig(savefile, dpi=400)
    print(savefile + ' 파일이 저장되었습니다.')
# end def FileSave():

import seaborn as sns

displ = mpg['displ'] # Series
print('displ.unique()')
print(displ.unique())

import numpy as np
print(np.max(displ.unique()))
print(np.min(displ.unique()))

'''
러그(rug) 플롯은 데이터 위치를 x축 위에 작은 선분(rug)으로 나타내어 실제 데이터들의 위치를 보여주는 함수입니다.
'''
'''
다음 예제는 커널 밀도 그래프와 함께 러그(rug) 플롯을  같이 출력해주는 예시입니다.
'''
ax = plt.subplots()
sns.kdeplot(data=mpg, x="displ")
ax = sns.rugplot(data=mpg, x="displ")
ax.set_title('엔진 크기에 따른 Rug Plot/Kde 곡선')
ax.set_xlabel('엔진 크기')
ax.set_ylabel('값')
FileSave()

ax = plt.subplots()
sns.scatterplot(data=mpg, x='displ', y='cty')
ax = sns.rugplot(data=mpg, x='displ', y='cty')
ax.set_title('산점도와 Rug Plot')
ax.set_xlabel('엔진 크기')
ax.set_ylabel('주행 마일수')
FileSave()

ax = plt.subplots()
ax = sns.histplot(data=mpg, x='displ', kde=True, bins=30)
ax.set_title('엔진 크기 히스토그램')
ax.set_xlabel('엔진 크기')
ax.set_ylabel('')
FileSave()

print("mpg['drv'].unique()")
print(mpg['drv'].unique())

label_dict = {'f':'전륜 구동', '4':'사륜 구동', 'r':'후륜 구동'}

def setLabel(x):
    return label_dict[x]

idx = 0 # 색상 구분을 위한 카운터 변수
mpg['drv'] = mpg['drv'].apply(setLabel)

print('구동 방식별 갯수')
result = mpg.groupby('drv')['drv'].count()
print(result)

plt.figure()

ax = sns.countplot(x="drv", data=mpg)
ax.set_title('구동 방식별 개수')
ax.set_xlabel('구동 방식')
ax.set_ylabel('')
FileSave()

print("mpg['cyl'].unique()")
print(mpg['cyl'].unique())

print('실린더 개수')
result = mpg.groupby('cyl')['cyl'].count()
print(result)

plt.figure()
ax = sns.countplot(x="cyl", data=mpg)
ax.set_title('실린더수에 따른 countplot')
ax.set_xlabel('실린더 수')
ax.set_ylabel('')
FileSave()

plt.figure()
ax = sns.jointplot(x="displ", y='hwy', data=mpg)
FileSave()

plt.figure()
ax = sns.jointplot(x="displ", y='hwy', data=mpg, kind="kde")
FileSave()

plt.figure()
newmpg = mpg.loc[:, ['displ', 'cyl', 'cty', 'hwy']]
ax = sns.pairplot(data=newmpg)
FileSave()

plt.figure()
ax = sns.pairplot(data=newmpg, hue="cyl", markers=["o", "s", "D", "p"])
FileSave()

mpg_size = mpg.pivot_table(
    index="drv", columns="cyl", aggfunc="size", fill_value=0)
print('mpg_size')
print(mpg_size)

plt.figure()
# ax = sns.heatmap(mpg_size, annot=True, fmt="d")
ax = sns.heatmap(mpg_size, annot=True, fmt=".2g")
ax.set_title('Heatmap 그리기')
ax.set_xlabel('실린더의 수')
ax.set_ylabel('구동 방식')
FileSave()

plt.figure()
ax = sns.barplot(x="drv", y='hwy', data=mpg, errwidth=0)
ax.set_title('구동 방식에 따른 주행 마일수(barplot))')
ax.set_xlabel('구동 방식')
ax.set_ylabel('주행 마일수')
FileSave()

plt.figure()
ax = sns.barplot(x='drv', y='hwy', hue='cyl', data=mpg, errwidth=0)
ax.set_title("구동 방식에 따른 주행 마일수(hue='cyl'))")
ax.set_xlabel('구동 방식')
ax.set_ylabel('주행 마일수')
FileSave()

print('구동 방식에 따른 주행 마일수의 통계치 정보')
mygrouping = mpg.groupby('drv')['hwy']
print(mygrouping.describe())

plt.figure()
ax = sns.boxplot(x='drv', y='hwy', data=mpg)
ax.set_title('구동 방식에 따른 주행 마일수(boxplot)')
ax.set_xlabel('구동 방식')
ax.set_ylabel('주행 마일수')
FileSave()

plt.figure()
ax = sns.boxplot(x='drv', y='hwy', hue='cyl', data=mpg)
ax.set_title('구동 방식에 따른 주행 마일수(boxplot)')
ax.set_xlabel('구동 방식')
ax.set_ylabel('주행 마일수')
FileSave()

plt.figure()
ax = sns.violinplot(x='drv', y='hwy', data=mpg)
ax.set_title('구동 방식에 따른 주행 마일수(violinplot)')
ax.set_xlabel('구동 방식')
ax.set_ylabel('주행 마일수')
FileSave()

plt.figure()
ax = sns.violinplot(x='drv', y='hwy', hue='cyl', data=mpg)
ax.set_title('구동 방식에 따른 주행 마일수(violinplot)')
ax.set_xlabel('구동 방식')
ax.set_ylabel('주행 마일수')
FileSave()

# UserWarning: 5.7% of the points cannot be placed;
# you may want to decrease the size of the markers or use stripplot.
STRIP_SIZE = 3

plt.figure()
ax = sns.stripplot(x='drv', y='hwy', data=mpg, jitter=True, size=STRIP_SIZE)
ax.set_title('구동 방식에 따른 주행 마일수(stripplot)')
ax.set_xlabel('구동 방식')
ax.set_ylabel('주행 마일수')
FileSave()

plt.figure()
ax = sns.stripplot(x='drv', y='hwy', hue='cyl', data=mpg, jitter=True, size=STRIP_SIZE)
ax.set_title('구동 방식에 따른 주행 마일수(stripplot)')
ax.set_xlabel('구동 방식')
ax.set_ylabel('주행 마일수')
plt.legend(loc=1)
FileSave()

plt.figure()
ax = sns.stripplot(x='drv', y='hwy', hue='cyl',
              data=mpg, jitter=True, dodge=True, size=STRIP_SIZE)
ax.set_title('stripplot(dodge=True)')
ax.set_xlabel('구동 방식')
ax.set_ylabel('주행 마일수')
FileSave()

plt.figure()
ax = sns.swarmplot(x='drv', y='hwy', data=mpg)
ax.set_title('Swarm Plot')
ax.set_xlabel('구동 방식')
ax.set_ylabel('주행 마일수')
FileSave()

plt.figure()
ax = sns.swarmplot(x='drv', y='hwy', hue='cyl', data=mpg)
ax.set_title("Swarm Plot(hue='cyl')")
ax.set_xlabel('구동 방식')
ax.set_ylabel('주행 마일수')
plt.legend(loc=1)
FileSave()

plt.figure()
ax = sns.swarmplot(x='drv', y='hwy', hue='cyl', data=mpg, dodge=True)
ax.set_title("Swarm Plot(dodge=True)")
ax.set_xlabel('구동 방식')
ax.set_ylabel('주행 마일수')
FileSave()

import numpy as np
plt.figure()
sns.boxplot(x='displ', y='drv', data=mpg, whis=np.inf)
ax = sns.stripplot(x='displ', y='drv', data=mpg, jitter=True, color="0.4")
ax.set_title("Boxplot과 Strip Plot로 표현")
ax.set_xlabel('엔진 크기')
ax.set_ylabel('구동 방식')
FileSave()

plt.figure()
sns.violinplot(x='drv', y='hwy', data=mpg, inner=None)
ax = sns.swarmplot(x='drv', y='hwy', data=mpg, color="0.9")
ax.set_title("Violin plot과 Swarm Plot로 표현")
ax.set_xlabel('구동 방식')
ax.set_ylabel('엔진 크기')
FileSave()

def sine_plot(mytitle):
    plt.figure()
    plt.rc('font', family='Malgun Gothic')
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i))
    plt.title('스타일 : ' + mytitle)
    FileSave()

styles = ["ticks", "darkgrid", "whitegrid"]
for one in styles:
    sns.set_style(one)
    sine_plot(one)

print('finished')