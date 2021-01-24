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
    def __init__(self, size_w=10, size_h=5):
        """
        変数をセット
        """

        # 基点の縦横の数
        self.size_w = size_w
        self.size_h = size_h

        # 道生成クラスのインスタンス生成
        # 引数 max_col は画像出力時のパレット数
        self.roadcreater =RoadCreater(max_col=size_w)
    
    def generate_maze(self, is_show=True, unit=10, delay=100):
        """
        道生成
        Returns
        -------
        field: 2d numpuy.array
            生成した迷路
        """

        # 実際のサイズ
        n_w = 2 * self.size_w + 1
        n_h = 2 * self.size_h + 1

        # 全てを壁で満たす
        # 0: 壁, 1: 基点当、表示用
        # 2以上:通路
        field = np.zeros((n_h, n_w), dtype=np.uint8)

        road_id = 2 # 0, 1は壁としている、道のid は2から

        # 道の生成
        x, y = (1, 1)
        field = self.roadcreater.create_road(
            field, x, y, id=road_id, is_show=is_show,
            delay=delay, unit=unit,
            )
        
        # 道のid を全て1に変換
        field_out = field.copy()
        field_out[field > 0] = 1

        return field_out


class RoadCreater:
    def __init__(self, max_col=10):
        self.render = Render(max_col=max_col)

    def create_road(self, field, x, y, id=2, is_show=True, delay=100, unit=10):
        # 開始地点を (x, y) とする
        # (A)
        # distance=2 分離れた4方向のセルの状態を調べる
        # 道を作れる方向（0 or 1）があったら、その中からランダムに選び道を作り、(x, y)を更新。
        # 道が作れなくなるまで、(A)からを繰り返す。
        #         
        # (B)
        # distance 分もどり、その(x2, y2)からこの関数を開始する。id = id + 1とする
        # 開始地点に戻るまで(B)を繰り返す。

        self.field = field.copy()
        self.delay = delay
        self.unit = unit
        
        # 開始地点が0 or 1 なら id とする
        if self.field[y, x] < 2:
            self.field[y, x] = id
        
        # xy の履歴 (B)で使用
        xy_history = [(x, y)]

        # (A)
        while True:
            # x, y の周囲を調べて、道が伸ばせるならば伸ばす
            res, x, y, fields = self.extend_road(self.field, x, y, id, distance=2)
            self.field = fields[-1]

            if res == 'stretched':
                # 道を伸ばした場合は、xy を記録して繰り返す
                xy_history.append((x, y))

                # 描画
                if is_show is True:
                    if self.delay > 50:
                        for ff in fields:
                            self.render.draw(ff, delay=self.delay, unit=self.unit)
                    else:
                        self.render.draw(fields[-1], delay=self.delay, unit=self.unit)
                continue

            # 行き止まりに来たら
            # ループから抜けて行き止まり対策
            if res == 'deadend':
                break
        
        # (B)
        # 履歴を一つずつもどった地点から、行き止まりまで道を伸ばす
        xy_history.pop(-1)
        xy_history.reverse()
        for xy in xy_history:
            x = xy[0]
            y = xy[1]
            self.create_road(
                self.field, x, y, id=id + 1, is_show=is_show,
                delay=self.delay, unit=self.unit,
                )
        
        return self.field

    def extend_road(self, field, x, y, id, distance=2):
        dd = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        ]
        n_h, n_w = field.shape[:2]
        next_field = field.copy()
        pre_field = field.copy()
        dd_res = [''] * 4
        for id_dir in range(4):
            x1 = x + dd[id_dir][0] * 2
            y1 = y + dd[id_dir][1] * 2
            x0 = x + dd[id_dir][0]
            y0 = y + dd[id_dir][1]

            if x1 < 0 or n_w <= x1:
                dd_res[id_dir] = 'out'
                continue

            if y1 < 0 or n_h <= y1:
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
            fields = [pre_field, next_field]

            return res, next_x, next_y, fields

        # 行き止まり
        res = 'deadend'
        next_x = None
        next_y = None
        fields = [pre_field, next_field]
        return res, next_x, next_y, fields


class Render():
    """
    画像生成、表示
    """
    def __init__(self, max_col=10):
        self.max_col = max_col # 色の種類の上限
        self.colorpalette = sns.color_palette(
            "hls", n_colors = self.max_col,
            )

    def draw(
        self,
        field,
        unit=10,
        is_show=True, delay=0, xy=None, unicol=None,
        ):

        val = field
        
        if xy is not None:
            val[xy[1], xy[0]] = 1

        max_val = np.max(val)
        val = val.astype(dtype=np.uint8)
        val = cv2.resize(
            val,
            dsize=(0, 0),
            fx=unit, fy=unit,
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
            img_r[val > 0] = unicol[0]
            img_g[val > 0] = unicol[1]
            img_b[val > 0] = unicol[2]

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
        self.render = Render(max_col=10)

    def solve_maze(self, is_show=False, delay=100, unit=10):
        self.map = self.field.copy()
        self.render.draw(
            self.map, is_show=is_show, unit=unit, delay=delay,
            )
        return self.map
    
   
        
if __name__ == '__main__':

    ptype = 'anime'
    # ptype = 'solve'

    if ptype == 'anime':
        ws = [10, 15, 20, 40]
        hs = [6, 9, 12, 24]
        us = [40, 27, 20, 10]
        ds = [50, 50, 10, 1]
        for w, h, u, d in zip(ws, hs, us, ds):
            for i in range(1):
                maze = Maze(size_w=w, size_h=h)
                field = maze.generate_maze(
                    is_show=True, unit=u, delay=d,
                    )
                maze.roadcreater.render.draw(
                    field, unit=u, delay=2000, unicol=(200, 200, 200),
                    )
        maze.roadcreater.render.draw(
            field, unit=u, delay=0, unicol=(200, 200, 200),
            )
        
    if ptype == 'solve':
        w, h, u, d = 10, 6, 40, 10
        # w, h, u, d = 20, 12, 27, 10
        maze = Maze(size_w=w, size_h=h)
        field = maze.generate_maze(is_show=True, unit=u, delay=d)
        print(field)
        maze.roadcreater.render.draw(field, unit=u, delay=0, unicol=(200, 200, 200))

        # solver = MazeSolver(field, start=(1, 1), goal=(1, 2))
        # map = solver.solve_maze(is_show=True, unit=u, delay=60)
        # solver.render.draw(map, unit=u, delay=0)




