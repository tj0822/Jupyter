import numpy as np
import matplotlib.pyplot as plt
###############################################################################
plt.rc('font', family='Malgun Gothic')
cnt, PNG, UNDERBAR = 0, '.png', '_'
CHART_NAME = 'barChartExam'
filename = './../data/주요발생국가주간동향(4월2째주).csv'
###############################################################################
import pandas as pd

data = pd.read_csv(filename, index_col='국가')

print(data.columns)
print('-'*30)

chartdata = data['4월06일']
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

    YTICKS_INTERVAL = 50000

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
        value = format(y[idx], ',') + '건'  # 예시 : 60건
        ratioval = '%.1f%%' % (ratio[idx])  # 예시 : 20.0%
        # 그래프의 위에 "건수" 표시
        plt.text(x=idx, y=y[idx] + 1, s=value, horizontalalignment='center')
        # 그래프의 중간에 비율 표시
        plt.text(x=idx, y=y[idx] / 2, s=ratioval, horizontalalignment='center')

    # 평균 값을 수평선으로 그리기
    meanval = y.mean()
    print(meanval)
    print('-' * 40)

    average = '평균 : %d건' % meanval
    plt.axhline(y=meanval, color='r', linewidth=1, linestyle='dashed')
    plt.text(x=y.size - 1, y=meanval + 200, s=average, horizontalalignment='center')

    global cnt
    cnt = cnt + 1
    savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
    plt.savefig(savefile, dpi=400)
    print(savefile + ' 파일이 저장되었습니다.')
# def MakeBarChart01
###############################################################################
'''
그래프에 대한 색상을 지정하는 리스트입니다.
예시에서 "w"는 흰색이라서 제외하도록 합니다.
'''
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

mycolor = colors[0:len(chartdata)]

'''
데이터 프레임을 이용하여 막대 그래프를 그려 주는 함수를 호출합니다.
'''
MakeBarChart01(x=chartdata.index, y=chartdata, color=mycolor, xlabel='국가명', ylabel='발생건수', title='국가별 코로나 발생 건수')
###############################################################################
'''
데이터 프레임을 사용하여 막대 그래프를 그려 주는 함수입니다.
'''

def MakeBarChart02(chartdata, rotation, title, ylim=None, stacked=False, yticks_interval = 10000):
    plt.figure()
    # 범례에 제목을 넣으려면 plot() 메소드의 legend 옵션을 사용해야 합니다.
    chartdata.plot(kind='bar', rot=rotation, title=title, legend=True, stacked=stacked)

    plt.legend(loc='best')

    print('chartdata')
    print(chartdata)

    if stacked == False :
        # max(chartdata.max())은 항목들 값 중에서 최대 값을 의미합니다.
        maxlim = (int(max(chartdata.max()) / yticks_interval) + 1) * yticks_interval
        print('maxlim : ', maxlim)
        values = np.arange(0, maxlim + 1, yticks_interval)
        plt.yticks(values, ['%s' % format(val, ',') for val in values])
    else : # 누적 막대 그래프
        # 국가별 누적 합인 chartdata.sum(axis=1))의 최대 값에 대한 연산이 이루어 져야 합니다.
        maxlim = (int(max(chartdata.sum(axis=1)) / yticks_interval) + 1) * yticks_interval
        print('maxlim : ', maxlim)
        values = np.arange(0, maxlim + 1, yticks_interval)
        plt.yticks(values, ['%s' % format(val, ',') for val in values])

    # y축의 상하한 값이 주어 지는 경우에만 설정합니다.
    if ylim != None :
        plt.ylim(ylim)

    global cnt
    cnt = cnt + 1
    savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
    plt.savefig(savefile, dpi=400)
    print(savefile + ' 파일이 저장되었습니다.')
# def MakeBarChart02
###############################################################################
'''
이번 예시에서는 특정 국가별 특정 일자에 대한 다변량 막대 그래프를 그려 보고자 합니다.
'''
data = pd.read_csv(filename, index_col='국가')
print(data.columns)
print('-' * 30)

COUNTRY = ['프랑스', '중국', '영국', '이란']
WHEN = ['4월06일', '4월07일', '4월08일']
data = data.loc[COUNTRY, WHEN]

print(data)
print('-'*30)

data.index.name = '국가명'
data.columns.name = '일자'

MakeBarChart02(chartdata=data, rotation=0, title='국가별 일별 발생 건수' )
###############################################################################
# 전치 프레임을 그래프로 그려 보기
dataT = data.T
print(dataT)
print('-' * 40)

MakeBarChart02(chartdata=dataT, rotation=0, title='일별 국가별 발생 건수')
###############################################################################
ymax = dataT.sum(axis=1)
ymaxlimit = ymax.max() + 10

MakeBarChart02(chartdata=data, rotation=0, title='국가별 일별 발생 건수(누적)', ylim=[0, ymaxlimit], stacked=True, yticks_interval=50000)
###############################################################################
data = pd.read_csv(filename, index_col='국가')

three = [item for item in data.index if item in ['프랑스', '영국', '중국']]
print(three)

data = data.loc[three]

print(data)
print('-' * 30)

column_names = data.columns.tolist()
print('column_names')
print(column_names)

# 국가별 numpy 배열을 저장하고 있는 사전
chartdata = {}

for row in data.index:
    # print(data.loc[row])
    # print(type(row))
    chartdata[row] = data.loc[row].values

print('chartdata')
print(chartdata)

def MakeBarChart03(chartdata, column_names):
    """
    Parameters
    ----------
    chartdata : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *column_names*.
    column_names : list of str
        The category labels.
    """
    labels = list(chartdata.keys())
    data = np.array(list(chartdata.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(column_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(column_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    global cnt
    cnt = cnt + 1
    savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
    plt.savefig(savefile, dpi=400)
    print(savefile + ' 파일이 저장되었습니다.')

    return fig, ax
# end def MakeBarChart03

MakeBarChart03(chartdata, column_names)
###############################################################################
def MakeBarChart04(chartdata, suptitle):
    fig, axes = plt.subplots(nrows=2, ncols=1)  # 2행 1열

    chartdata.plot(kind='bar', ax=axes[0], rot=0, alpha=0.7)

    # color='m'은 자홍색
    chartdata.plot(kind='barh', ax=axes[1], color='m', alpha=0.7)

    fig.suptitle(suptitle)  # sup : super

    global cnt
    cnt = cnt + 1
    savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
    plt.savefig(savefile, dpi=400)
    print(savefile + ' 파일이 저장되었습니다.')
# end def MakeBarChart04
###############################################################################
data = pd.read_csv(filename, index_col='국가')

only_usa = [item for item in data.index if item in ['미국']]
# print(only_usa)

data = data.loc[only_usa].T

print(data)
print(type(data))
print('-'*30)

MakeBarChart04(chartdata=data, suptitle='서브 플로팅')
###############################################################################
# 엑셀처럼 Table이 존재하는 Bar Chart 그리기
data = pd.read_csv(filename, index_col='국가')

print(data.columns)
print('-' * 30)

COUNTRY = ['스페인', '프랑스', '중국', '영국', '이란']
WHEN = ['4월06일', '4월07일', '4월08일', '4월09일', '4월10일']
data = data.loc[COUNTRY, WHEN]

print('data')
print(data)
print('-'*30)

# rows : 테이블에 보이는 행 색인 내용
rows = [x for x in data.index]
print('rows')
print(rows)
print('-'*30)

# columns : 테이블에 보이는 열 색인 내용
columns = [x for x in data.columns]
print('columns')
print(columns)
print('-'*30)

print('데이터 최대 값 :', max(data.max()))
print('-'*30)

n_rows = len(data) # 행 수
print('n_rows :', n_rows)
print('-'*30)

LEFT_MARGIN = 0.3
index = np.arange(len(columns)) + LEFT_MARGIN
print('index :', index)
print('-'*30)

bar_width = 1 - 2 * LEFT_MARGIN # 막대 그래프의 너비

# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.zeros(len(columns))
print('y_offset :', y_offset)
print('-'*30)

# Plot bars and create text labels for the table
cell_text = [] # 표에 들어 가는 텍스트 내용
plt.figure()

for row in data.index:
    print('data[row]')
    chartdata = data.loc[row].tolist()
    print(chartdata)

    # bottom
    plt.bar(index, chartdata, bar_width, bottom=y_offset, label=row)

    # y_offset에는 열 단위로 누적된 값이 들어 갑니다.
    y_offset = y_offset + chartdata
    # y_offset = chartdata
    print('y_offset')
    print(y_offset)

    cell_text.append([format(x, ',') for x in chartdata])
    # cell_text.append([format(x, ',') for x in y_offset])
# end for

cell_text.reverse()
rows = [rows[idx] for idx in range(len(rows) - 1, -1, -1) ]

# Add a table at the bottom of the axes
print('cell_text : ', cell_text)
print('rows : ', rows)
# print('colors : ', colors)
print('columns : ', columns)
the_table = plt.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='bottom')

plt.legend(loc='best')
# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.ylabel("발생 건수")

# values : y축의 눈금의 상한 값과 간격 지정하기
YTICKS_INTERVAL = 50000 # 단위 눈금 간격
maxlim = (int(y_offset.max()/YTICKS_INTERVAL)+1)*YTICKS_INTERVAL
print(maxlim)

values = np.arange(0, maxlim, YTICKS_INTERVAL)

plt.yticks(values, ['%s' % format(val, ',') for val in values])
plt.xticks([])
plt.title('테이블이 있는 막대 그래프')

cnt += 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')
###############################################################################
print('finished')