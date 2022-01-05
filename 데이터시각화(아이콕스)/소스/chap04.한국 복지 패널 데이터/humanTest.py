import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#################################################################
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
plt.rcParams["font.size"] = 12
plt.rcParams['xtick.labelsize'] = 12.
plt.rcParams['ytick.labelsize'] = 12.
plt.rcParams['axes.unicode_minus'] = False

CHART_NAME = 'seabornWelfare_exam'
cnt, PNG, UNDERBAR = 0, '.png', '_'
filename = './../data/human_info.csv'
# filename = './../data/welfare_python.csv'
################################################################
import pandas as pd
welfare = pd.read_csv(filename, encoding='cp949')
############################################################
def FileSave():
    global cnt
    cnt += 1
    savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
    plt.savefig(savefile, dpi=400)
    print(savefile + ' 파일이 저장되었습니다.')
# end def FileSave():
#################################################
def newAge(x):
    if x < 40:
        return '청년'
    elif x >= 40 and x < 50:
        return '중년'
    else :
        return '노년'

welfare['ageg'] = welfare['age'].apply(newAge)
################################################################
print('결혼 유무에 따른 빈도')
result = welfare.groupby('marriage')['marriage'].count()
print(result)

ORDERING = ['결혼', '이혼', '미혼']
plt.figure()
plt.title('결혼 유무에 따른 빈도')

import seaborn as sns
sns.countplot(x='marriage', data=welfare, order=ORDERING)
FileSave()

print('marriage vs religion 빈도')
result = welfare.groupby(['marriage', 'religion'])['marriage'].count()
print(result)

plt.figure()
plt.title('결혼 유무에 vs 종교 유무 빈도')
sns.countplot(x='marriage', hue='religion', data=welfare, order=ORDERING)
FileSave()
################################################################
x = welfare['age']

plt.figure() # kde(kernel density)
plt.title('kde와 histogram')
sns.distplot(x, rug=False, hist=True, kde=True, label='asdf')
FileSave()

pivot = welfare.pivot_table(index='gender', columns='marriage', values='age')
print('pivot_table을 이용한 시각화')
print(pivot)

plt.figure()
plt.title('결혼 유무와 성별에 대한 히트맵')
sns.heatmap(data=pivot, annot=True)
FileSave()

# correlation(상관관계)를 시각화
# corr() 함수는 데이터의 상관 관계를 보여줍니다.
cor = welfare.corr()
print('상관 관계 시각화')
print(cor)

plt.figure()
plt.title('상관 관계 시각화')
sns.heatmap(data=cor, annot=True, cmap="YlGnBu")
FileSave()
##########################################################
newwelfare = welfare.loc[:, ['income', 'age', 'marriage']]

plt.figure()
plt.title('hue 옵션으로 특성 구분')
sns.pairplot(data=newwelfare, hue='marriage')
FileSave()
#############################################################
print("welfare['age'].describe()")
print(welfare['age'].describe())

plt.figure()
plt.title('나이에 대한 바이올린 그래프')
sns.violinplot(x='age', data=welfare)
FileSave()
##########################################################
plt.figure()
plt.title('')
sns.lmplot(x='age', y='income', hue='marriage', height=8, data=welfare)
FileSave()
#############################################################
plt.figure()
plt.title('컬러 팔레트 적용')
sns.relplot(x='age', y='income', hue='gender', row='ageg', col='marriage', palette='CMRmap_r', data=welfare)
FileSave()
##########################################################
plt.figure()
plt.title('기본 jointplot')
sns.jointplot(x='age', y='income', height=8, data=welfare)
FileSave()

ax = plt.subplots()
ax = sns.barplot(x='income', y='gender', hue='religion', data=welfare, errwidth=0)
ax.set_title('소득에 따른 성별 그래프')
ax.set_ylabel('gender')
ax.set_xlabel('income')
FileSave()

print('finished')