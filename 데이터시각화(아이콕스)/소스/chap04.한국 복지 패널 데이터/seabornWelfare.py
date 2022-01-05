import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
###############################################################################
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
plt.rcParams["font.size"] = 12
plt.rcParams['xtick.labelsize'] = 12.
plt.rcParams['ytick.labelsize'] = 12.
plt.rcParams['axes.unicode_minus'] = False

CHART_NAME = 'seabornWelfare'
cnt, PNG, UNDERBAR = 0, '.png', '_'
filename = './../data/welfare_python.csv'
###############################################################################
import pandas as pd
welfare = pd.read_csv(filename, encoding='utf-8')

print(welfare.columns)

### 데이터 전처리 ###############################################################
welfare.loc[welfare['gender'] == 1 , ['gender']] = '남성'
welfare.loc[welfare['gender'] == 2 , ['gender']] = '여성'

print('\n# 나이 컬럼은 존재하지 않으므로 생일 컬럼을 이용하여 산술 연산합니다.')
thisyear = 2020
welfare['age'] = thisyear - welfare['birth'] + 1

def setMarriage( x ):
    if x == 1 :
        return '결혼'
    elif x == 3 :
        return '이혼'
    else :
        return '무응답' # 결측치

# 결혼 : 숫자 1이면 결혼, 3이면 이혼, 이외에는 결측치로 처리
welfare['marriage'] = welfare['marriage'].apply(setMarriage)

print('\n# 월급 결측치 개수 구하기 before')
print(sum(welfare['income'].isnull()))

welfare.loc[welfare['income'].isnull(), 'income'] = welfare['income'].mean()

print('\n# 월급 결측치 개수 구하기 after')
print(sum(welfare['income'].isnull()))

def setReligion_txt( x ):
    if int(x) == 1 :
        return '있슴'
    else :
        return '없슴'

print("welfare['religion'].unique()")
print(welfare['religion'].unique())


welfare['religion'] = welfare['religion'].apply(setReligion_txt)

job_file = './../data/welfare_job.csv'
jobframe = pd.read_csv(job_file, encoding='cp949')

print("welfare['code_job'].unique()")
print(welfare['code_job'].unique())

print('\n# merge() 함수의 left_on과 right_on 사용하기')
welfare = pd.merge( welfare, jobframe, left_on='code_job', right_on='code_job')
print(welfare)

print("welfare['code_religion'].unique()")
print(welfare['code_religion'].unique())

def setReligion_txt( x ):
    if int(x) == 1 :
        return '서울'
    elif int(x) == 2 :
        return '수도권'
    elif int(x) == 3:
        return '부산/경남/울산'
    elif int(x) == 4 :
        return '대구/경북'
    elif int(x) == 5 :
        return '대전/충남'
    elif int(x) == 6 :
        return '강원/충북'
    elif int(x) == 7 :
        return '광주/전남/전북/제주도'

welfare['code_religion'] = welfare['code_religion'].apply(setReligion_txt)

# 청년_young(30세 미만), 중년_middle(30세 이상), 노년_old(60세 이상)
def newAge(x):
    if x < 30:
        return '청년'
    elif x >= 30 and x < 60:
        return '중년'
    else :
        return '노년'

welfare['ageg'] = welfare['age'].apply(newAge)

print(welfare[['age', 'ageg']].head())


col_mapping = {'gender':'성별', 'birth':'생일', 'marriage':'결혼 유무', 'religion':'종교 유무', 'code_job':'직업 코드', 'income':'소득', 'code_religion':'지역구', 'age':'나이', 'job':'직업', 'ageg':'연령대'}
welfare = welfare.rename(columns = col_mapping)

welfare.to_csv('welfareClean.csv', index=False, encoding='cp949')
###############################################################################
print(welfare.columns)

print(welfare.head(10))

print(welfare.describe())

print(welfare['결혼 유무'].unique())

print(welfare['종교 유무'].unique())

print(welfare['지역구'].unique())

print(welfare['직업'].unique())

print(welfare['연령대'].unique())

###############################################################################
def FileSave():
    global cnt
    cnt += 1
    savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
    plt.savefig(savefile, dpi=400)
    print(savefile + ' 파일이 저장되었습니다.')
# end def FileSave():
###################################################################################################
print('결혼 유무 빈도')
result = welfare.groupby('결혼 유무')['결혼 유무'].count()
print(result)

ORDERING = ['결혼', '이혼', '무응답']
plt.figure()
plt.title('결혼 유무 빈도')

import seaborn as sns
sns.countplot(x='결혼 유무', data=welfare, order=ORDERING)
FileSave()


print('결혼 유무 vs 종교 유무 빈도')
result = welfare.groupby(['결혼 유무', '종교 유무'])['결혼 유무'].count()
print(result)

plt.figure()
plt.title('결혼 유무 vs 종교 유무 빈도')
sns.countplot(x='결혼 유무', hue='종교 유무', data=welfare, order=ORDERING)
FileSave()

plt.figure()
plt.title('막대 테두리에 색상 넣기')
sns.countplot(x='결혼 유무', hue='종교 유무', data=welfare, order=ORDERING, linewidth=1, edgecolor=sns.color_palette("dark", 3))
FileSave()

plt.figure()
plt.title('가로 막대로 그리기')
sns.countplot(y='결혼 유무', hue='종교 유무', data=welfare, order=ORDERING)
FileSave()

plt.figure()
plt.title('색상 팔레트 설정')
sns.countplot(x='결혼 유무', hue='종교 유무', palette='Paired', data=welfare, order=ORDERING)
FileSave()
###################################################################################################
x = welfare['나이']

plt.figure()
plt.title('rugplot과 kde')
sns.distplot(x, rug=True, hist=False, kde=True)
FileSave()

plt.figure() # kde(kernel density)
plt.title('kde와 histogram')
sns.distplot(x, rug=False, hist=True, kde=True, label='asdf')
FileSave()

plt.figure()
plt.title('가로로 표현하기')
sns.distplot(x, vertical=True)
FileSave()

plt.figure()
plt.title('컬러 바꾸기')
sns.distplot(x, color="m")
FileSave()
###################################################################################################
"""
매개 변수 values는 집계하고자(보고자) 하는 컬럼 이름 또는 리스트를 명시합니다.
즉, cell에 들어갈 항목을 여기에 명시합니다.
예시에서는 '나이'에 대한 정보를 사용합니다.
매개 변수 index는 행에 보여줄 데이터 이름을 의미하는 데, '성별'을 사용합니다.
매개 변수 columns는 컬럼에 보여줄 데이터 이름 또는 리스트를 명시하는 데 '결혼 유무'를 사용합니다.
"""

pivot = welfare.pivot_table(index='성별', columns='결혼 유무', values='나이')
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
###################################################################################################
newwelfare = welfare.loc[:, ['직업 코드', '소득', '나이', '결혼 유무']]

plt.figure()
plt.title('기본 pairplot')
sns.pairplot(data=newwelfare)
FileSave()

plt.figure()
plt.title('hue 옵션으로 특성 구분')
sns.pairplot(data=newwelfare, hue='결혼 유무')
FileSave()

plt.figure()
plt.title('컬러 팔레트 적용')
sns.pairplot(data=newwelfare, hue='결혼 유무', palette="rainbow")
FileSave()

plt.figure()
plt.title('사이즈 적용')
sns.pairplot(data=newwelfare, hue='결혼 유무', palette="rainbow", height=5,)
FileSave()
###################################################################################################
print("welfare['나이'].describe()")
print(welfare['나이'].describe())

plt.figure()
plt.title('나이에 대한 바이올린 그래프')
sns.violinplot(x='나이', data=welfare)
FileSave()

print('x, y축을 지정해 줌으로써 바이올린을 분할하여 비교 분포를 볼 수 있습니다.')
plt.figure()
plt.title('비교 분포 확인')
sns.violinplot(x='성별', y='나이', data=welfare)
FileSave()

plt.figure()
plt.title('가로형 violinplot')
sns.violinplot(y='성별', x='나이', data=welfare)
FileSave()

plt.figure()
plt.title('hue 옵션으로 분포 비교')
sns.violinplot(x='종교 유무', y='나이', hue='성별', data=welfare, palette="muted")
FileSave()
# ###################################################################################################
# plt.figure()
# plt.title('split 옵션으로 바이올린 합치기')
# sns.violinplot(x='성별', y='나이', hue='지역구', data=welfare, palette="muted", split=True)
# FileSave()
###################################################################################################
plt.figure()
plt.title('기본 lmplot')
sns.lmplot(x='나이', y='소득', height=8, data=welfare)
FileSave()
#
# 6-2. hue 옵션으로 다중 선형관계 그리기
# 아래의 그래프를 통하여 비흡연자가, 흡연자 대비 좀 더 가파른 선형관계를 가지는 것을 볼 수 있습니다.
plt.figure()
plt.title('')
sns.lmplot(x='나이', y='소득', hue='결혼 유무', height=8, data=welfare)
FileSave()
#
# 6-3. col 옵션을 추가하여 그래프를 별도로 그려볼 수 있습니다
# 또한, col_wrap으로 한 줄에 표기할 column의 갯수를 명시할 수 있습니다.
plt.figure()
plt.title('')
sns.lmplot(x='나이', y='소득', hue='결혼 유무', col='성별', col_wrap=2, height=8, data=welfare)
FileSave()
###################################################################################################
plt.figure()
plt.title('기본 relplot')
sns.relplot(x='나이', y='소득', hue='성별', data=welfare)
FileSave()

plt.figure()
plt.title('col 옵션으로 그래프 분할')
sns.relplot(x='나이', y='소득', hue='성별', col='결혼 유무', data=welfare)
FileSave()
#
# 7-3. row와 column에 표기할 데이터 column 선택
plt.figure()
plt.title('')
sns.relplot(x='나이', y='소득', hue='성별', row='연령대', col='결혼 유무', data=welfare)
FileSave()

plt.figure()
plt.title('컬러 팔레트 적용')
sns.relplot(x='나이', y='소득', hue='성별', row='연령대', col='결혼 유무', palette='CMRmap_r', data=welfare)
FileSave()
###################################################################################################
plt.figure()
plt.title('기본 jointplot')
sns.jointplot(x='나이', y='소득', height=8, data=welfare)
FileSave()

#선형 관계를 위한 회귀선(regression line)은 kind='reg'을 사용하면 됩니다.
plt.figure()
plt.title('산점도와 histogram')
sns.jointplot(x='나이', y='소득', height=8, data=welfare, kind="reg")
FileSave()

plt.figure()
plt.title('hex 밀도 보기')
sns.jointplot(x='나이', y='소득', height=8, data=welfare, kind="hex")
FileSave()

# 8-4. 등고선 모양으로 밀집도 확인하기
# kind='kde' 옵션으로 데이터의 밀집도를 보다 부드러운 선으로 확인할 수 있습니다.
plt.figure()
plt.title('등고선 모양으로 밀집도 확인')
sns.jointplot(x='나이', y='소득', height=8, data=welfare, kind="kde", color="b")
FileSave()

print(welfare[['소득', '나이']].describe())

ax = plt.subplots()
ax = sns.barplot(x='성별', y='소득', data=welfare, errwidth=0)
ax.set_title('성별에 따른 소득 그래프')
ax.set_xlabel('성별')
ax.set_ylabel('소득')
FileSave()

ax = plt.subplots()
ax = sns.barplot(x='성별', y='소득', hue='종교 유무', data=welfare, errwidth=0)
ax.set_title('성별에 따른 소득 그래프')
ax.set_xlabel('성별')
ax.set_ylabel('소득')
FileSave()

ax = plt.subplots()
ax = sns.barplot(x='소득', y='성별', hue='종교 유무', data=welfare, errwidth=0)
ax.set_title('소득에 따른 성별 그래프')
ax.set_ylabel('성별')
ax.set_xlabel('소득')
FileSave()

ax = plt.subplots()
ax = sns.boxplot(x='성별', y='소득', data=welfare)
ax.set_title('성별에 따른 소득 상자 수염 그래프')
ax.set_xlabel('성별')
ax.set_ylabel('소득')
FileSave()

newwelfare = welfare.loc[ welfare['소득'] <= 400 ]
ax = plt.subplots()
ax = sns.boxplot(x='성별', y='소득', data=newwelfare)
ax.set_title('성별에 따른 소득 상자 수염 그래프')
ax.set_xlabel('성별')
ax.set_ylabel('소득')
FileSave()

kde, ax = plt.subplots()
ax = sns.kdeplot(data=newwelfare['소득'],
                 data2=newwelfare['나이'],
                 shade=True)
ax.set_title('소득에 따른 나이(이차원 밀집도 shade=True)')
ax.set_xlabel('소득')
ax.set_ylabel('나이')
FileSave()

kde, ax = plt.subplots() 
ax = sns.kdeplot(data=newwelfare['소득'],
                 data2=newwelfare['나이'],
                 shade=False)
ax.set_title('소득에 따른 나이(이차원 밀집도 shade=False)')
ax.set_xlabel('소득')
ax.set_ylabel('나이')
FileSave()

print('finished')