# 型注釈のないcython関数

def calculate_z(maxiter, zs, cs):
    """ジュリア集合の更新ルールを使って出力リストを計算する"""
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while n < maxiter and abs(z) < 2:
            z = z * z + c
            n += 1
        output[i] = n
    return output
