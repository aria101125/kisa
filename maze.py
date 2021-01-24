"""
迷路生成プログラム
穴掘り法

参考
http://www5d.biglobe.ne.jp/stssk/maze/make.html
この理論だけだとまだ未完成。
機能追加して完成

maze2.py
生成過程を表示できるようにする
"""

import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pdb



class Maze():
    """
    迷路生成クラス
    """
    def __init__(self, size_w=10, size_h=5, unit=10, delay=10):
        """
        変数のセット
        """
        self.max_col = int(size_w / 2)# 色の種類の上限
        self.size_w = size_w
        self.size_h = size_h
        self.unit = unit
        self.delay = delay
        self.colorpalette = sns.color_palette("hls", n_colors = self.max_col)
    
    def generate_maze(self, is_show=True):
        """
        Returns
        -------
        field: 2d numpuy.array
            生成した迷路
        """

        # 実際のサイズ
        self.n_w = 2 * self.size_w + 1
        self.n_h = 2 * self.size_h + 1

        # 全てを壁で満たす
        # 0: 壁, 1: 基点, 2以上:通路
        self.field = np.zeros((self.n_h, self.n_w), dtype=np.uint8)

        # 基点となるセルの座標をランダムな順番でリストに生成
        self.base_xys = self._get_base()

        # 起点を1にする
        for xy in self.base_xys:
            self.field[xy[1], xy[0]] = 1

        # 全ての起点をランダムな順番で巡回し、
        # 道がなかったら道を作る
        result = 0
        # for j, xy in enumerate(self.base_xys):
        road_id = 2 # 
        while True:
            idxs = np.where(self.field == 1)
            num = len(idxs[0])
            if num is 0:
                break
            idx = np.random.randint(0, num)
            # 0: 壁、2以上:通路
            x = idxs[1][idx]
            y = idxs[0][idx]
            if self.field[y, x] > 1:
                # すでに通路になっていたら次のポイントへ
                continue
            
            # x, y を開始点として、自分以外の壁にぶつかるまで
            # 道を伸ばす
            ret = self._create_road(x, y, id=road_id, is_show=is_show)
            road_id += 1
            result += (ret == False)
        
        field = self.get_maze()
        
        return field

    def get_maze(self):
        field = self.field.copy()
        field[field > 0] = 1
        return field
    
    def _create_road(self, x, y, id=2, is_show=True):  # (A)
        # 行き止まりになるまで道を伸ばす
        # ペン先を1つ戻した位置から、 ret = _create_road() で再帰
        # 道の根元まで全てもどったら終了
        # 
        field = self.field.copy()
        if field[y, x] < 2:
            field[y, x] = id
        xy_history = [(x, y)]
        while True:
            # x, y から道を伸ばす
            res, field, x, y, pre_field = self.find_load_from(field, x, y, id)
            self.field = field

            if res == 'stretched':
                # 道を伸ばした場合は、繰り返す
                if is_show is True:
                    if self.delay > 50:
                        self.render(field=pre_field, delay=self.delay)
                    self.render(delay=self.delay)
                xy_history.append((x, y))
                continue

            # 行き止まりに来たら
            # ループから抜けて行き止まり対策
            if res == 'deadend':
                break

        # 履歴を一つずつもどった地点から、行き止まりまで道を伸ばす
        xy_history.pop(-1)
        xy_history.reverse()
        for xy in xy_history:
            x = xy[0]
            y = xy[1]
            self._create_road(x, y, id=id + 1, is_show=is_show)

    def find_load_from(self, field, x, y, id):
        dd = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        ]
        next_field = field.copy()
        pre_field = field.copy()
        dd_res = [''] * 4
        for id_dir in range(4):
            x1 = x + dd[id_dir][0] * 2
            y1 = y + dd[id_dir][1] * 2
            x0 = x + dd[id_dir][0]
            y0 = y + dd[id_dir][1]

            if x1 < 0 or self.n_w <= x1:
                dd_res[id_dir] = 'out'
                continue

            if y1 < 0 or self.n_h <= y1:
                # はみ出したら次の方向を試す
                dd_res[id_dir] = 'out'
                continue

            if field[y1, x1] in (0, 1):
                # 壁
                dd_res[id_dir] = 'wall'
                continue

            # 道
            dd_res[id_dir] = 'same_id'

        # 壁が周りにあったらそこに道を伸ばしてもどる
        ids_dir = [i for i, x in enumerate(dd_res) if x == 'wall']
        if len(ids_dir) > 0:
            res = 'stretched'
            id_dir = random.sample(ids_dir, 1)[0]
            x1 = x + dd[id_dir][0] * 2
            y1 = y + dd[id_dir][1] * 2
            x0 = x + dd[id_dir][0]
            y0 = y + dd[id_dir][1]
            next_field[y1, x1] = id
            next_field[y0, x0] = id
            next_x = x1
            next_y = y1

            pre_field[y0, x0] = id  # アニメーション用の途中状態

            return res, next_field, next_x, next_y, pre_field

        # 行き止まり
        res = 'deadend'
        next_x = None
        next_y = None
        return res, next_field, next_x, next_y, pre_field

    def _get_base(self):
        xs = np.arange(1, self.n_w, 2)
        ys = np.arange(1, self.n_h, 2)
        xxs, yys = np.meshgrid(xs, ys)
        xys = np.vstack([xxs.reshape(-1), yys.reshape(-1)])
        xys = xys.T
        xys = xys.tolist()
        xys = random.sample(xys, k=len(xys))
        return xys


class Render(object):
    """
    画像生成、表示
    """
    @staticmethod
    def draw(
        self, field=None, 
        is_show=True, delay=0, xy=None, unicol=None,
        ):
        if field is None:
            val = self.field.copy()
        else:
            val = field
        
        if xy is not None:
            val[xy[1], xy[0]] = 1

        max_val = np.max(val)
        val = val.astype(dtype=np.uint8)
        val = cv2.resize(
            val,
            dsize=(0, 0),
            fx=self.unit, fy=self.unit,
            interpolation=cv2.INTER_NEAREST,
            )
        img_r = val.copy()
        img_g = val.copy()
        img_b = val.copy()

        if unicol is None:
            cols = {
                0: (50, 50, 50),  # wall
                1: (0, 100, 0), # base
            }
            for v in range(max_val + 1):
                if v < 2:
                    col = cols[v]
                else:
                    ic = v % self.max_col
                    col = np.array(self.colorpalette[ic]) * 255
                img_r[val == v] = col[0]
                img_g[val == v] = col[1]
                img_b[val == v] = col[2]
        else:
            img_r[val > 1] = unicol[0]
            img_g[val > 1] = unicol[1]
            img_b[val > 1] = unicol[2]

        h, w = val.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = img_b
        img[:, :, 1] = img_g
        img[:, :, 2] = img_r

        if is_show:
            cv2.imshow('img', img)
            INPUT = cv2.waitKey(delay) & 0xFF
            if INPUT == ord('q'):
                sys.exit()

        return img



class MazeSolver:
    def __init__(self, field, start, goal):
        self.field = field
        self.start_xy = start
        self.goal_xy = goal
        self.xy = self.start_xy

    def solve_maze(self, is_show=False):

        return field
    
    def render(self):
        img = None
        return img
    
        
if __name__ == '__main__':

    # ptype = 'anime'
    ptype = 'solve'

    if ptype == 'anime':
        ws = [10, 15, 20, 40]
        hs = [6, 9, 12, 24]
        us = [40, 27, 20, 10]
        ds = [100, 50, 10, 1]
        for w, h, u, d in zip(ws, hs, us, ds):
            for i in range(1):
                maze = Maze(size_w=w, size_h=h, unit=u, delay=d)
                maze.generate_maze()
                maze.render(delay=1000)
                maze.render(delay=2000, unicol=(200, 200, 200))
        maze.render(delay=0, unicol=(200, 200, 200))
    if ptype == 'solve':
        w, h, u, d = 10, 6, 40, 100
        # w, h, u, d = 20, 12, 27, 10
        maze = Maze(size_w=w, size_h=h, unit=u, delay=d)
        field = maze.generate_maze(is_show=False)
        print(field)
        # maze.render(delay=0, unicol=(200, 200, 200))
        # maze.render(delay=0)

        solver = MazeSolver(field)
        field = solver.solve_maze()
        solver.render(delay=0)



