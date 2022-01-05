import numpy as np
import pandas as pd

from matplotlib import cbook, colors as mcolors
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox, TransformedBbox, BboxTransformTo
###############################################################################
plt.rc('font', family='Malgun Gothic')
cnt, PNG, UNDERBAR = 0, '.png', '_'
CHART_NAME = 'ribbonBox_exam'
filename = './../data/연령별_실업율.csv'
###############################################################################
class RibbonBox:
    original_image = plt.imread(
        cbook.get_sample_data("Minduka_Present_Blue_Pack.png"))
    cut_location = 70
    b_and_h = original_image[:, :, 2:3]
    color = original_image[:, :, 2:3] - original_image[:, :, 0:1]
    alpha = original_image[:, :, 3:4]
    nx = original_image.shape[1]

    def __init__(self, color):
        rgb = mcolors.to_rgba(color)[:3]
        self.im = np.dstack(
            [self.b_and_h - self.color * (1 - np.array(rgb)), self.alpha])

    def get_stretched_image(self, stretch_factor):
        stretch_factor = max(stretch_factor, 1)
        ny, nx, nch = self.im.shape
        ny2 = int(ny*stretch_factor)
        return np.vstack(
            [self.im[:self.cut_location],
             np.broadcast_to(
                 self.im[self.cut_location], (ny2 - ny, nx, nch)),
             self.im[self.cut_location:]])
# end class RibbonBox:

class RibbonBoxImage(AxesImage):
    zorder = 1

    def __init__(self, ax, bbox, color, *, extent=(0, 1, 0, 1), **kwargs):
        super().__init__(ax, extent=extent, **kwargs)
        self._bbox = bbox
        self._ribbonbox = RibbonBox(color)
        self.set_transform(BboxTransformTo(bbox))

    def draw(self, renderer, *args, **kwargs):
        stretch_factor = self._bbox.height / self._bbox.width

        ny = int(stretch_factor*self._ribbonbox.nx)
        if self.get_array() is None or self.get_array().shape[0] != ny:
            arr = self._ribbonbox.get_stretched_image(stretch_factor)
            self.set_array(arr)

        super().draw(renderer, *args, **kwargs)
# end class RibbonBoxImage(AxesImage):

def main():
    fig, ax = plt.subplots()

    # data = pd.read_csv(filename, index_col='국가')
    data = pd.read_csv(filename, index_col='연도', encoding='cp949')

    koreadata = data.loc['2006년', '20대':'60세이상']
    print('koreadata.index')
    print(koreadata.index)

    chartdata = [koreadata[item] for item in koreadata.index]
    print('chartdata')
    print(chartdata) # 그리고자 하는 데이터

    xdata = np.arange(0, len(chartdata))
    print('xdata')
    print(xdata)

    box_colors = [
        (0.8, 0.2, 0.2),
        (0.2, 0.8, 0.2),
        (0.2, 0.2, 0.8),
        (0.7, 0.5, 0.8),
        (0.3, 0.8, 0.7),
    ]

    for x, h, bc in zip(xdata, chartdata, box_colors):
        bbox0 = Bbox.from_extents(x - 0.4, 0., x + 0.4, h)
        bbox = TransformedBbox(bbox0, ax.transData)
        # 리본 이미지 넣기
        ax.add_artist(RibbonBoxImage(ax, bbox, bc, interpolation="bicubic"))
        # 상단의 수치 데이터를 콤마 유형으로 표시
        ax.annotate('%s' % format(h,','), (x, h), va="bottom", ha="center")

    ax.set_xlim(xdata[0] - 0.5, xdata[-1] + 0.5)
    ax.set_ylim(0, 10)

    myxticks = [item for item in koreadata.index]
    ax.set_xticks(xdata)
    ax.set_xticklabels(myxticks) # x축에 놓을 문자열
    ax.set_title('연령대별 실업률(2006년)')

    # 배경 색상을 지정합니다.
    background_gradient = np.zeros((2, 2, 4))
    background_gradient[:, :, :3] = [1, 1, 0]
    background_gradient[:, :, 3] = [[0.1, 0.3], [0.3, 0.5]]  # alpha channel
    ax.imshow(background_gradient, interpolation="bicubic", zorder=0.1,
              extent=(0, 1, 0, 1), transform=ax.transAxes, aspect="auto")

    global cnt
    cnt += 1
    savefile = CHART_NAME + UNDERBAR + str(cnt).zfill(2) + PNG
    plt.savefig(savefile, dpi=400)
    print(savefile + ' 파일이 저장되었습니다.')

    print('finished')
# end main()

main()