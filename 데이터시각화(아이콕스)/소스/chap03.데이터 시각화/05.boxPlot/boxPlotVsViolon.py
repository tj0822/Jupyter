"""
A good general reference on boxplots and their history can be found
here: http://vita.had.co.nz/papers/boxplots.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

filename = './../data/tips.csv'
myframe = pd.read_csv(filename, encoding='utf-8', index_col=0)

DINNER, LUNCH = 'Dinner', 'Lunch'

frame01 = myframe.loc[myframe['time'] == DINNER, 'total_bill']
frame01.index.name = DINNER

frame02 = myframe.loc[myframe['time'] == LUNCH, 'total_bill']
frame02.index.name = LUNCH

chartdata = [np.array(frame01), np.array(frame02)]
print('chartdata')
print(chartdata)

# plot violin plot
axs[0].violinplot(chartdata,
                  showmeans=False,
                  showmedians=True)
axs[0].set_title('Violin plot')

axs[1].boxplot(chartdata) # plot box plot
axs[1].set_title('Box plot')

# adding horizontal grid lines
for ax in axs:
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(chartdata))])
    ax.set_xlabel('점심과 저녁 팁')
    ax.set_ylabel('observed data')

# add x-tick labels
plt.setp(axs, xticks=[y + 1 for y in range(len(chartdata))],
         xticklabels=[DINNER, LUNCH])


PNG, UNDERBAR = '.png', '_'
cnt = 0

cnt += 1
savefile = 'boxPlotVsViolon' + UNDERBAR + str(cnt).zfill(2) + PNG
plt.savefig(savefile, dpi=400)
print(savefile + ' 파일이 저장되었습니다.')

print('finished')