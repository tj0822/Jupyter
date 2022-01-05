import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')
cnt, PNG, UNDERBAR = 0, '.png', '_'
CHART_NAME = 'histogramExam'
filename = './../data/tips.csv'
plt.rcParams['axes.unicode_minus'] = False
num_bins = 30

tips = pd.read_csv(filename, encoding='utf-8')

fig, ax = plt.subplots()

x = tips['total_bill'] # Series
print(type(x))
# print('x')
# print(x)
# print('-'*30)

# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=True)

ax.set_title('Histogram of Total Bill')
ax.set_xlabel('Frequency')
ax.set_ylabel('Total Bill')

import numpy as np

mu = x.mean()  # 평균
print('mu :', mu)

sigma = x.std() # 표준 편차
print('sigma :', sigma)

y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
ax.plot(bins, y, '--')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()

cnt += 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')

# 만약 최소값/최대값을 바꾸거나 계급 구간의 개수를 늘리고 싶다면 range와 bins를 통해 직접 설정할 수 있다.
fig, ax = plt.subplots()
plt.hist(x, range=(5, 40), bins=num_bins)

cnt += 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')

humanfile = './../data/human_height.csv'

human = pd.read_csv(humanfile, encoding='utf-8')

print('\n#별개의 데이터에 대한 histogram 서브 플로팅')
fig, axes = plt.subplots(nrows=1, ncols=2)
giant = human['giant']
dwarf = human['dwarf']

axes[0].hist(giant, range=(210, 290), bins=20, alpha=0.6)
axes[1].hist(dwarf, range=(100, 180), bins=20, alpha=0.6)

axes[0].set_title('거인국의 키(height)')
axes[1].set_title('소인국의 키(height)')

cnt = cnt + 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')

print('\n#2개의 histogram 같이 그리기')
fig, axes = plt.subplots()

axes.hist(giant, bins=20, alpha=0.6)
axes.hist(dwarf, bins=20, alpha=0.6)

cnt = cnt + 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')

print('\n#stack histogram 그리기')
fig, axes = plt.subplots()
man = human['man']
woman = human['woman']

x = np.array([man, woman]).T
print(x)
print(x.shape) # (1000, 2)

axes.hist(x, bins=num_bins, density=False, histtype='bar', stacked=True)
axes.set_title('누적 히스토그램')

cnt = cnt + 1
savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')

print('finished')