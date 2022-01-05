import matplotlib
import matplotlib.pyplot as plt
###############################################################################
plt.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False
cnt, PNG, UNDERBAR = 0, '.png', '_'
CHART_NAME = 'pieChartExam'
filename = './../data/주요발생국가주간동향(4월2째주).csv'
###############################################################################
import pandas as pd

data = pd.read_csv(filename, index_col='국가')
print(data.columns)
print('-'*30)

my_concern = [item for item in data.index if item in ['독일', '프랑스', '중국', '영국']]
print(my_concern)

data = data.loc[my_concern]

chartdata = data['4월06일']

print(chartdata)
print('-'*30)

print(type(chartdata))
print('-'*30)


mylabel = chartdata.index

print(mylabel)
print('-'*30)

mycolors = ['blue', '#6AFF00', 'yellow', '#FF003C']

plt.figure()

plt.pie(chartdata, labels=mylabel, shadow=False, explode=(0, 0.05, 0, 0),
        colors=mycolors, autopct='%1.2f%%', startangle=90, counterclock=False)

plt.grid(True)
plt.legend(loc=4)
plt.xlabel('국가명')
# plt.ylabel("발생 건수")
plt.title('주요 국가 발생 건수')

cnt += 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')
###############################################################################
import numpy as np
fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

def getLabelFormat(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.2f}%\n({:d} 명)".format(pct, absolute)

wedges, texts, autotexts = ax.pie(chartdata, autopct=lambda pct: getLabelFormat(pct, chartdata),
                                  textprops=dict(color="w"))

ax.legend(wedges, mylabel,
          title="국가명",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")

ax.set_title("주요 국가 발생 건수")

cnt += 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')
###############################################################################
print('도우넛 그래프를 그려 봅니다.')
data = pd.read_csv(filename, index_col='국가')

fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

COUNTRY = ['독일', '프랑스', '중국', '영국', '이탈리아']

data = data.loc[COUNTRY, ['4월06일']]
print(data.values.flatten())
print('-'*30)

wedges, texts = ax.pie(data.values.flatten(), wedgeprops=dict(width=0.5), startangle=-40)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    print('ang : ', ang)
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))

    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]

    connectionstyle = "angle,angleA=0,angleB={}".format(ang)

    kw["arrowprops"].update({"connectionstyle": connectionstyle})

    ax.annotate(COUNTRY[i], xy=(x, y), xytext=(1.3*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, **kw)

ax.set_title("도우넛 그래프")

cnt += 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
# plt.legend(loc='best')
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')
###############################################################################
# The most straightforward way to build a pie chart is to use the
# :meth:`pie method <matplotlib.axes.Axes.pie>`
#
# In this case, pie takes values corresponding to counts in a group.
# We'll first generate some fake data, corresponding to three groups.
# In the inner circle, we'll treat each number as belonging to its
# own group. In the outer circle, we'll plot them as members of their
# original 3 groups.
#
# The effect of the donut shape is achieved by setting a ``width`` to

# the pie's wedges through the *wedgeprops* argument.

print('주요 국가별 중첩 파이 그래프를 그려 봅니다.')
fig, ax = plt.subplots()

data = pd.read_csv(filename, index_col='국가')

print(data)
print('-'*30)

COUNTRY = ['독일', '프랑스', '중국', '영국', '이탈리아']
my_concern = [item for item in data.index if item in COUNTRY]
print(my_concern)

data = data.loc[my_concern]

filtered_data = data[['4월06일', '4월07일']]

print(filtered_data)
print('-'*30)

print(filtered_data.index.values)
print('-'*30)

totallist = [] # 차트를 그릴 중첩 데이터
for key in filtered_data.index.values :
       imsi = filtered_data.loc[key].values
       totallist.append([item for item in imsi])

chartdata = np.array(totallist)
print('chartdata : \n', chartdata)

color_su = len(COUNTRY) # 색상의 개수
cmap = plt.get_cmap("tab20c")
outer_colors = cmap(np.arange(color_su)*4)

inner_colors = cmap(np.arange(2*color_su))
print('inner_colors :', inner_colors)
print('outer_colors :', outer_colors)

cum_sum = chartdata.sum(axis=1) # 누계
print('cum_sum : ', cum_sum)

# 숫자가 적을 수록 가운데 비어 있는 원이 커집니다.
INNER_VACANT_CIRCLE_SIZE = 0.3

# OUTER_PCTDISTANCE : 비율을 보여주는 위치를 지정하는 데, 원점에서의 거리를 지정하면 됩니다.
OUTER_PCTDISTANCE = 0.85
# edgecolor='w', 'None'
ax.pie(cum_sum, radius=1, colors=outer_colors,
       wedgeprops=dict(width=INNER_VACANT_CIRCLE_SIZE, edgecolor='w'),
       labels=COUNTRY, autopct='%.2f%%', pctdistance=OUTER_PCTDISTANCE)

INNER_PCTDISTANCE = 0.75

ax.pie(chartdata.flatten(), radius=1-INNER_VACANT_CIRCLE_SIZE, colors=inner_colors,
       wedgeprops=dict(width=INNER_VACANT_CIRCLE_SIZE, edgecolor='w'),
       autopct='%.2f%%', pctdistance=INNER_PCTDISTANCE)

ax.set(aspect="equal", title='주요 국가별 중첩 파이 그래프')

cnt += 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
# plt.legend(loc='best')
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')
###############################################################################
# figure와 축(axis) 객체
fig = plt.figure(figsize=(9, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
fig.subplots_adjust(wspace=0)

data = pd.read_csv(filename, index_col='국가')

print(data)
print('-'*30)

COUNTRY = ['독일', '프랑스', '중국'] # 관심 국가 목록
my_concern = [item for item in data.index if item in COUNTRY]
print('my_concern : ', my_concern)
print('-'*30)

data = data.loc[my_concern]

when = ['4월06일', '4월07일', '4월08일'] # 관심 일자 목록
filtered_data = data[when] # 파이 차트에 그려질 데이터

print('filtered_data : \n', filtered_data)
print('-'*30)

pieData = filtered_data.sum(axis=1).values # 국가별 총합

barData = filtered_data.loc['독일'].values
barData = barData/sum(barData)

print('pieData : ', pieData)
print('-'*30)

print('barData : ', barData)
print('-'*30)

# pie chart 관련 변수 리스트
explode = [0 for idx in range(len(pieData))]
explode[0] = 0.05
print('explode : ', explode)
print('-'*30)

# rotate so that first wedge is split by the x-axis
# 막대 그래프를 우측에 그릴 것이므로, 시작 각도는 90도로 지정하고 counterclock=False의 값으로 지정하면 좋습니다.
STARTANGLE = 90
print('STARTANGLE : ', STARTANGLE)
print('-'*30)
ax1.pie(pieData, autopct='%1.1f%%', startangle=STARTANGLE,
        labels=COUNTRY, explode=explode, counterclock=False)

# bar chart 관련 변수
xpos = 0
bottom = 0
width = .2
colors = [[.1, .3, .5], [.1, .3, .3], [.1, .3, .7]]
colors = ['r', 'g', 'b']

for j in range(len(barData)):
    height = barData[j]
    ax2.bar(xpos, height, width, bottom=bottom, color=colors[j])
    ypos = bottom + ax2.patches[j].get_height() / 2
    bottom += height
    ax2.text(xpos, ypos, "%.2f%%" % (ax2.patches[j].get_height() * 100),
             ha='center', fontsize=8, color='w')

ax2.set_title("'" + COUNTRY[0] + "'의 일자별 비율")
ax2.legend((when))
ax2.axis('off')
ax2.set_xlim(- 2.5 * width, 2.5 * width)

# use ConnectionPatch to draw lines between the two plots
# get the wedge data
theta1, theta2 = ax1.patches[0].theta1, ax1.patches[0].theta2
center, r = ax1.patches[0].center, ax1.patches[0].r
bar_height = sum([item.get_height() for item in ax2.patches])

LINE_WIDTH = 2 # 연결선의 두께

from matplotlib.patches import ConnectionPatch

# 상단의 연결선
x = r * np.cos(np.pi / 180 * theta2) + center[0]
y = r * np.sin(np.pi / 180 * theta2) + center[1]
con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                      xyB=(x, y), coordsB=ax1.transData)
con.set_color([0, 0, 0])
con.set_linewidth(LINE_WIDTH)
ax2.add_artist(con)

# 하단의 연결선
x = r * np.cos(np.pi / 180 * theta1) + center[0]
y = r * np.sin(np.pi / 180 * theta1) + center[1]
con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                      xyB=(x, y), coordsB=ax1.transData)
con.set_color([0, 0, 0])
ax2.add_artist(con)
con.set_linewidth(LINE_WIDTH)

cnt += 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
# plt.legend(loc='best')
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')
###############################################################################
print('finished')
