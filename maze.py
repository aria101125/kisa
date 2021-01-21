"""
迷路生成プログラム
穴掘り法

参考
http://www5d.biglobe.ne.jp/stssk/maze/make.html
"""

import numpy as np
import random
import cv2

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

        # ４辺のセルを残して壁で満たす
        # self.field[1:self.n_h-1, 1:self.n_w-1] = 1

        # 基点となるセルの座標をランダムな順番でリストに生成
        self.base_xys = self._get_base()

        # self._get_base()のチェック
        """
        for i, xy in enumerate(self.base_xys):
            x = xy[0]
            y = xy[1]
            self.field[y, x] = i + 1
        """
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
            ret = self._create_road(x, y, id=j+1)
            result += ret == False
        return result
        
    def _create_road(self, x, y, id=1):
        # id 以外の道にぶつかるまで道を伸ばす
        # id = 1 のときは、進めなくなるまで道を伸ばす
        self.field[y, x] = id
        isCont = True
        dd = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        )
        c = 0
        while isCont:
            c += 1
            if c > 2000:
                print('繰り返しが多すぎます')
                pdb.set_trace()
            isTerminal = True  # 行き止まり
            ids_dir = random.sample([0, 1, 2, 3], 4)
            for id_dir in ids_dir:
                # 4方向をランダムな順番でサーチ---------------
                x0 = x + dd[id_dir][0]
                y0 = y + dd[id_dir][1]
                x1 = x + dd[id_dir][0] * 2
                y1 = y + dd[id_dir][1] * 2

                if x1 < 0 or self.n_w <= x1:
                    # はみ出したら次の方向を試す
                    continue 

                if y1 < 0 or self.n_h <= y1:
                    # はみ出したら次の方向を試す
                    continue 

                if self.field[y1, x1] > 0:
                    # 今の方向に道があったら次の方向を試す
                    continue
                else:
                    # 道がない方向を発見
                    # その方向に道を作って伸ばした先を基点とし、
                    # forから抜ける
                    self.field[y1, x1] = id
                    self.field[y0, x0] = id
                    x = x1
                    y = y1
                    isTerminal = False
                    break
            # ----------------------------------------------
            # ここまで来て、isTreminal == True だったら
            # どの方向にも進めないということ
            # isTreminal == False なら、id の方向に進んだあと
            
            if isTerminal is True:
                # どの方向にも道があった場合、
                if id == 1:
                    # id == 1ならこれで終了
                    return True

                # id > 1 の場合、id が異なる方向に道をつなげてwhileから抜ける
                ids_dir = random.sample([0, 1, 2, 3], 4)
                for id_dir in ids_dir:
                    x0 = x + dd[id_dir][0]
                    y0 = y + dd[id_dir][1]
                    x1 = x + dd[id_dir][0] * 2
                    y1 = y + dd[id_dir][1] * 2
                    if x1 < 0 or self.n_w <= x1:
                        continue 
                    if y1 < 0 or self.n_h <= y1:
                        continue 
                    if self.field[y1, x1] != id and self.field[y1, x1] != 0:
                        # print('%d は%dにつながりました' %(id, self.field[y1, x1]))
                        self.field[y0, x0] = id
                        return True
                print('%d はつなげられませんでした' % id)
                return False


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
    maze = Maze(size_w=30, size_h=10, unit=10)
    for i in range(100):
        ret = maze.generate_maze()
        if ret == 0:
            break

    # print(maze.field)
    print(ret)
    mm = maze.render()
