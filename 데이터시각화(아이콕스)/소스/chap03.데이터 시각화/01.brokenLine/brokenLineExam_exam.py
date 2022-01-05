import pandas as pd
import matplotlib.pyplot as plt
###############################################################################
plt.rc('font', family='Malgun Gothic')
cnt, PNG, UNDERBAR = 0, '.png', '_'
CHART_NAME = 'brokenLineExam_연습_문제'
filename = './../data/연령별_실업율.csv'
# filename = './../data/주요발생연도주간동향(4월2째주).csv'
###############################################################################
data = pd.read_csv(filename, index_col='연도', encoding='cp949')
print(data.columns)


# YEAR = ['2001년', '2002년', '2003년', '2004년', '2005년', '2006년', '2007년', '2008년', '2009년', '2010년']
YEAR = ['2001년', '2002년', '2003년', '2004년', '2005년']
AGEG = ['20대', '30대', '40대', '50대', '60세이상']
chartdata = data.loc[YEAR, AGEG]

chartdata = chartdata.T

chartdata.plot(title='SomeTitie', figsize=(10, 6), legend = True, marker='o', rot=0)

# plt.grid(True)
plt.xlabel('년령대')
plt.ylabel('연도명')
plt.title('년령대별 연도명 꺽은 선')

cnt += 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')
