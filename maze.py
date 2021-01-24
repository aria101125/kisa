"""
迷路生成、迷路解きプログラム
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
        self.roadcreater =RoadCreater(
            max_col=size_w,
            mode='create',
            )
    
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
        # 0: 壁
        # 1以上: 道
        field = np.zeros((n_h, n_w), dtype=np.uint8)

        road_id = 2

        # 道の生成
        x = 2* np.random.randint(0, self.size_w) + 1
        y = 2* np.random.randint(0, self.size_h) + 1
        field = self.roadcreater.create_road(
            field, x, y, id=road_id, is_show=is_show,
            delay=delay, unit=unit,
            )
        
        # 道のid を全て1に変換
        field_out = field.copy()
        field_out[field > 0] = 1

        return field_out


class MazeSolver:
    """
    迷路を解くクラス
    """
    def __init__(self, field, start, goal):
        self.field = field.copy()
        self.start_xy = start
        self.goal_xy = goal
        self.xy = self.start_xy
        max_col = int(self.field.shape[1] / 2)
        self.roadcreater =RoadCreater(
            max_col=max_col,
            mode='solve',
            goal_xy=goal,
            start_xy=start,
            )

    def solve_maze(self, is_show=True, delay=100, unit=10):
        """
        迷路の壁が0、通路が1
        通路に沿って進む、分岐点に来たらゴールが近くなる方を選ぶ

        迷路生成クラスと同じ、RoadCreaterクラスを内部で使用
        """

        map = self.field.copy()
        x, y = self.start_xy
        road_id = 2
        map[y, x] = road_id
        map_out = self.roadcreater.create_road(
            map, x, y, id=road_id, is_show=is_show,
            delay=delay, unit=unit,
            )
        return map_out
   
        
class RoadCreater:
    """
    指定したidの領域を、道を分岐させながら成長させるクラス
    """
    def __init__(self, max_col=10, mode='create', goal_xy=None, start_xy=None):
        self.render = Render(
            max_col=max_col, mode=mode, 
            goal_xy=goal_xy, start_xy=start_xy,
            )
        self.reach_goal = False
        self.mode = mode
        self.goal_xy = goal_xy
        self.start_xy = start_xy

    def create_road(
        self, field, x, y, id=1, is_show=True, delay=100, unit=10,
        ):
        # mode 'create'
        # 開始地点を (x, y) とする
        # (A)
        # 2分離れた4方向のセルの状態を調べる
        # 道を作れる方向（0 or 1）があったら、その中からランダムに選び道を作り、(x, y)を更新。
        # 道が作れなくなるまで、(A)からを繰り返す。
        #         
        # (B)
        # 2つ分もどり、その(x2, y2)からこの関数を開始する。id = id + 1とする
        # 開始地点に戻るまで(B)を繰り返す。
        # 
        # mode 'solve'

        if self.reach_goal is True:
            return field

        self.field = field.copy()
        self.delay = delay
        self.unit = unit
        
        # 開始地点が0なら id とする
        if self.field[y, x] == 0:
            self.field[y, x] = id

        # xy の履歴 (B)で使用
        xy_history = [(x, y)]

        # (A)
        while True:
            # x, y の周囲を調べて、道が伸ばせるならば伸ばす
            res, x, y, fields = self.extend_road(
                self.field, x, y, id,
                )
            self.field = fields[-1]

            if res == 'stretched':
                # 道を伸ばした場合は、xy を記録して繰り返す
                xy_history.append((x, y))

                # 描画
                if is_show is True:
                    if self.delay > 50:
                        for ff in fields:
                            self.render.draw(
                                ff, delay=self.delay, unit=self.unit,
                                )
                    else:
                        self.render.draw(
                            fields[-1], delay=self.delay, unit=self.unit,
                            )

                # ゴール判定
                if self.mode == 'solve':
                    if x == self.goal_xy[0] and y == self.goal_xy[1]:
                        # ゴールに辿り着いたら終了
                        self.reach_goal = True
                        return self.field

                continue


            # 行き止まりに来たら
            # ループから抜けて(B)へ
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

    def extend_road(
        self, field, x, y, id,
        ):
        """
        id_wall の数値のある場所に道を作っていく
        mode='create': 迷路生成時
        mode='solve': 迷路を解く時
        """
        if self.mode == 'create':
            id_wall = 0
        elif self.mode == 'solve':
            id_wall = 1
        else:
            raise ValueError('modeが違います')

        dd = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        ]
        n_h, n_w = field.shape[:2]
        next_field = field.copy()
        pre_field = field.copy()

        # 4方向の状態を調べる
        # 'out', 'goal', 'wall', 'road'
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

           
            if (self.mode == 'create' and field[y1, x1] == id_wall) or \
                (self.mode == 'solve' and field[y0, x0] == id_wall): 
                # 壁(道を伸ばせる)
                dd_res[id_dir] = 'wall'
                continue
                

            # 他の道がある（道を伸ばせない）
            dd_res[id_dir] = 'road'

        # 壁が周りにあったらそこに道を伸ばしてもどる
        ids_dir = [i for i, x in enumerate(dd_res) if x == 'wall']
        if len(ids_dir) > 0:
            res = 'stretched'
            if self.mode == 'create':
                # 生成時にはランダム
                id_dir = random.sample(ids_dir, 1)[0]
            elif self.mode == 'solve':
                # 解いているときにはゴールに近くなる方を選ぶ
                dist = []
                for i, id_dir in enumerate(ids_dir):
                    x0 = x + dd[id_dir][0]
                    y0 = y + dd[id_dir][1]
                    dist.append((x0 - self.goal_xy[0])**2 + (y0 - self.goal_xy[1]) ** 2)
                iid = dist.index(min(dist))
                id_dir = ids_dir[iid]
            else:
                raise ValueError('modeが違います')

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
    def __init__(
        self, max_col=10, 
        goal_xy=None, start_xy=None, mode='create',
        ):
        self.max_col = max_col # 色の種類の上限
        self.goal_xy=goal_xy
        self.start_xy=start_xy
        self.mode=mode

        self.colorpalette = sns.color_palette(
            "hls", n_colors = self.max_col,
            )

    def draw(
        self,
        field,
        unit=10,
        is_show=True, delay=0,
        unicol=None,
        start_xy=None,
        goal_xy=None,
        ):

        if start_xy is not None:
            self.start_xy = start_xy
        if goal_xy is not None:
            self.goal_xy = goal_xy
    

        val = field
        
        max_val = np.max(val)
        val = val.astype(dtype=np.uint8)
        val = cv2.resize(
            val,
            dsize=(0, 0),
            fx=unit, fy=unit,
            interpolation=cv2.INTER_NEAREST,
            )
        
        #--- for video fix size
        """
        val_bak = np.ones((440, 840), dtype=np.uint8) * 80
        h, w = val.shape[:2]
        val_bak[:h, :w] = val
        val = val_bak
        """
        # ----


        img_r = val.copy()
        img_g = val.copy()
        img_b = val.copy()

        for v in range(max_val + 1):
            if self.mode == 'create':
                if v == 0: # 壁（道を伸ばす）
                    col = (80, 80, 80)
                elif v == 1: # 壁（道を伸ばす）
                    col = (255, 255, 255)
                else:
                    ic = v % self.max_col
                    col = np.array(self.colorpalette[ic]) * 255
            elif self.mode == 'solve':
                if v == 0: # 壁
                    col = (80, 80, 80)
                elif v == 1: # 道、道を伸ばす
                    col = (255, 255, 255)
                else:
                    ic = v % self.max_col
                    col = np.array(self.colorpalette[ic]) * 255
            else:
                raise ValueError('mode が違います')

            img_r[val == v] = col[0]
            img_g[val == v] = col[1]
            img_b[val == v] = col[2]

        h, w = val.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = img_b
        img[:, :, 1] = img_g
        img[:, :, 2] = img_r

        if self.start_xy is not None:
            x = int(self.start_xy[0] * unit + unit / 2)
            y = int(self.start_xy[1] * unit + unit / 2)
            r = int(0.45 * unit)
            col = (100, 100, 200)
            img = cv2.circle(img, (x, y), r, col, -1)

        if self.goal_xy is not None:
            x = int(self.goal_xy[0] * unit + unit / 2)
            y = int(self.goal_xy[1] * unit + unit / 2)
            r = int(0.45 * unit)
            col = (100, 200, 100)
            img = cv2.circle(img, (x, y), r, col, -1)

        if is_show:
            cv2.imshow('img', img)
            INPUT = cv2.waitKey(delay) & 0xFF
            if INPUT == ord('q'):
                sys.exit()

        return img


if __name__ == '__main__':

        prms = {
            0: (10, 5, 40, 100),
            1: (20, 10, 20, 50),
            2: (40, 20, 10, 1),
        }
        w, h, u, d = 20, 12, 10, 10

        # for video
        """
        cv2.imshow('img', np.ones((440, 840, 3), dtype=np.uint8) * 80)
        cv2.waitKey(0)
        """

        for i in range(10000):
            w, h, u, d = prms[i % len(prms)]
            maze = Maze(size_w=w, size_h=h)
            field = maze.generate_maze(is_show=True, unit=u, delay=d)
            f_h, f_w = field.shape[:2]
            start_xy = (1, 1)
            goal_xy = (f_w - 2, f_h - 2)

            img = maze.roadcreater.render.draw(
                field, unit=u, delay=1000, unicol=(255, 255, 255),
                start_xy=start_xy, goal_xy=goal_xy,
                )
            print(img.shape, 'size')

            solver = MazeSolver(
                field, start=start_xy, goal=goal_xy)
            map = solver.solve_maze(is_show=True, unit=u, delay=d)
            solver.roadcreater.render.draw(map, unit=u, delay=1000)




