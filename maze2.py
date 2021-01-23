"""
迷路生成プログラム
穴掘り法

参考
http://www5d.biglobe.ne.jp/stssk/maze/make.html
この理論だけだとまだ未完成。
機能追加して完成
"""

import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

import pdb

class Maze():
    """
    迷路生成クラス
    """
    def __init__(self, size_w=10, size_h=5, unit=10):
        """
        変数のセット
        """
        self.size_w = size_w
        self.size_h = size_h
        self.unit = unit
    
    def generate_maze(self):
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
        # 0: 壁、1以上:通路
        self.field = np.zeros((self.n_h, self.n_w), dtype=np.uint8)
        # import pdb; pdb.set_trace()
        # 基点となるセルの座標をランダムな順番でリストに生成
        self.base_xys = self._get_base()

        result = 0
        for j, xy in enumerate(self.base_xys):
            # ランダムな順番で一つの基点を決める
            # 0: 壁、1以上:通路
            x = xy[0]
            y = xy[1]
            if self.field[y, x] > 0:
                # すでに通路になっていたら次のポイントへ
                continue
            
            # x, y を開始点として、自分以外の壁にぶつかるまで
            # 道を伸ばす
            ret = self._create_road(x, y, id=j + 1)
            result += (ret == False)
        return result
    
    def _create_road(self, x, y, id=1):  # (A)
        # id 以外の道にぶつかるまで道を伸ばす
        # id = 1 のときは、進めなくなるまで道を伸ばす
        field = self.field.copy()
        field[y, x] = id
        xy_history = [(x, y)]
        while True:
            # x, y から道を伸ばす
            res, field, x, y = self.find_load_from(field, x, y, id)

            if res == 'connected':
                # 別な道につながった場合は終了
                self.field = field
                return True
            
            if res == 'stretched':
                # 道を伸ばした場合は、繰り返す
                xy_history.append((x, y))
                continue

            # 行き止まりに入り込んでしまったら、
            # ループから抜けて行き止まり対策
            if res == 'deadend':
                break

            raise ValueError('res が間違っています')
        
        if id == 1:
            self.field = field
            return True

        # 行き止まり対策
        while True:
            # ベースを一つもどる
            xy_history.pop(-1)
            if len(xy_history) == 0:
                # ベースが無くなったら異常終了
                return False

            x, y = xy_history[-1]
            ret = self._create_road(x, y, id=id)
            if ret is True:
                return True

    def find_load_from(self, field, x, y, id):
        dd = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        )
        next_field = field.copy()
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

            if field[y1, x1] == 0:
                # 壁
                dd_res[id_dir] = 'wall'
                continue

            if field[y1, x1] == id:
                # 自分
                dd_res[id_dir] = 'same_id'
                continue

            # 違うid
            dd_res[id_dir] = 'diff_id'

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
            return res, next_field, next_x, next_y

        ids_dir = [i for i, x in enumerate(dd_res) if x == 'diff_id']
        if len(ids_dir) > 0:
            res = 'connected'
            id_dir = random.sample(ids_dir, 1)[0]
            x0 = x + dd[id_dir][0]
            y0 = y + dd[id_dir][1]
            next_field[y0, x0] = id
            next_x = None
            next_y = None
            return res, next_field, next_x, next_y

        res = 'deadend'
        next_x = None
        next_y = None
        return res, next_field, next_x, next_y


    def _get_base(self):
        xs = np.arange(1, self.n_w, 2)
        ys = np.arange(1, self.n_h, 2)
        xxs, yys = np.meshgrid(xs, ys)
        xys = np.vstack([xxs.reshape(-1), yys.reshape(-1)])
        xys = xys.T
        xys = xys.tolist()
        xys = random.sample(xys, k=len(xys))
        return xys

    def render(self, is_show=True):
        val = self.field.copy()
        val[val != 0] = 1
        maxv = np.max(val)
        minv = np.min(val)
        if maxv - minv > 0:
            val = (val.astype(float) - minv) / (maxv - minv) * 255
        val = val.astype(dtype=np.uint8)
        # val = 255 - val
        img = cv2.cvtColor(val, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(
            img,
            dsize=(0, 0),
            fx=self.unit, fy=self.unit,
            interpolation=cv2.INTER_NEAREST,
            )
        if is_show:
            # plt.pcolor(val)
            # plt.show()
            cv2.imshow('img', img)
            cv2.waitKey(0)
        return img

    def render_str(self, is_show=True):
        trans ={
            0: 'w',
            1: ' ',
        }
        other = 'w'

        h, w = self.field.shape[:2]
        lines=[]
        for iy in range(h):
            line = ''
            for ix in range(w):
                if self.field[iy, ix] in trans:
                    line += trans[self.field[iy, ix]]
                else:
                    line += other
            lines.append(line)
        
        if is_show:
            for l in lines:
                print(l)

        return lines

if __name__ == '__main__':
    maze = Maze(size_w=40, size_h=20, unit=10)
    ret = maze.generate_maze()
    print('繋がれなかった道の数', ret)
    mm = maze.render()
