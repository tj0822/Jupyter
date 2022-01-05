import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')
cnt, PNG, UNDERBAR = 0, '.png', '_'
CHART_NAME = 'histogramTest'
humanfile = './../data/human_height.csv'
plt.rcParams['axes.unicode_minus'] = False
num_bins = 50

human = pd.read_csv(humanfile, encoding='utf-8')

man = human['man']
woman = human['woman']

print(man.describe())

plt.figure(figsize=(8, 6))
plt.hist(man, bins=num_bins, alpha=0.7, facecolor='blue', label="남자", rwidth=0.9)

plt.xlabel("키", size=14)
plt.ylabel("빈도 수", size=14)
plt.title("단일 histogram")
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.75)
cnt = cnt + 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')

plt.figure(figsize=(8, 6))
plt.hist(man, bins=num_bins, alpha=0.5, label="남자")
plt.hist(woman, bins=num_bins, alpha=0.5, label="여자")

plt.xlabel("키", size=14)
plt.ylabel("빈도 수", size=14)
plt.title("다중 histogram 그래프")
plt.legend(loc='upper right')
cnt = cnt + 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')

print('finished')