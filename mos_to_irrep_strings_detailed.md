# mos_to_irrep_strings 関数詳細ドキュメント

## 概要

`mos_to_irrep_strings`関数は、PySCFの分子オブジェクトから各原子の基底関数をE3NN（Equivariant Neural Networks）の既約表現文字列に変換し、原子種の埋め込みインデックスを生成する重要な関数です。この関数は、量子化学計算で得られた基底関数情報を機械学習で使用可能な形式に変換します。

## 関数シグネチャ

```python
def mos_to_irrep_strings(mol) -> Tuple[Dict[int, int], Dict[int, str]]
```

## パラメータ詳細

### 入力パラメータ

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `mol` | `pyscf.gto.Mole` | PySCFの分子オブジェクト（基底関数情報を含む） |

### 出力

2つの辞書のタプルを返します：

1. `atom_type_embedding_per_atom`: `Dict[int, int]` - 各原子の原子種インデックス
2. `irrep_string_per_atom`: `Dict[int, str]` - 各原子のE3NN既約表現文字列

## 処理フロー

### 1. 基底関数ラベルの取得
```python
ao_labels = mol.ao_labels(fmt="%d,%2s,%s%-0s")
```
- PySCFから基底関数のラベルを取得
- フォーマット: `原子インデックス,原子種,軌道名`

### 2. 各基底関数の解析
各基底関数ラベルを解析して以下を抽出：
- **原子インデックス**: どの原子に属するか
- **原子種**: 元素の種類（C, H, Oなど）
- **軌道名**: 軌道の種類（1s, 2px, 3dxyなど）

### 3. 軌道タイプからE3NN既約表現への変換
```python
if orbital[1] == "s":
    irreps_string += "1x0e+"      # s軌道 → 1x0e (スカラー)
elif orbital[1] == "p" and orbital[2] == "x":
    irreps_string += "1x1o+"      # p軌道 → 1x1o (ベクトル)
elif orbital[1] == "d" and orbital[2] == "x" and orbital[3] == "y":
    irreps_string += "1x2e+"      # d軌道 → 1x2e (テンソル)
```

### 4. 原子種インデックスの生成
```python
atom_type_embedding_per_atom[atom_idx] = element_to_index[atom_type_string.strip()]
```

## 使用例

### 基本的な使用例

```python
import torch
import numpy as np
from pyscf import gto, scf
from egsmole.utils import hf_utils
import e3nn

# 分子の定義（メタン分子）
mol_string = """
    C   0.0  0.0  0.0
    H   1.0  1.0  1.0
    H  -1.0 -1.0  1.0
    H  -1.0  1.0 -1.0
    H   1.0 -1.0 -1.0
"""

# PySCFで分子オブジェクトを作成
mol = gto.Mole()
mol.atom = mol_string
mol.basis = "def2-svp"
mol.build()

# mos_to_irrep_strings関数を実行
atom_type_embedding_per_atom, irrep_string_per_atom = hf_utils.mos_to_irrep_strings(mol)

print(f"原子種埋め込み: {atom_type_embedding_per_atom}")
print(f"irrep文字列: {irrep_string_per_atom}")

# 各原子の詳細を確認
for atom_idx in range(mol.natm):
    atom_symbol = mol.atom_symbol(atom_idx)
    print(f"原子{atom_idx} ({atom_symbol}):")
    print(f"  原子種インデックス: {atom_type_embedding_per_atom[atom_idx]}")
    print(f"  irrep文字列: {irrep_string_per_atom[atom_idx]}")
    
    # irrep文字列を解析
    mo_irreps = e3nn.o3.Irreps(irrep_string_per_atom[atom_idx])
    print(f"  irrep次元: {mo_irreps.dim}")
```

### 入力データの例

#### 1. 基底関数ラベル（ao_labels）
```python
# メタン分子（CH4）の例
ao_labels = [
    "0, C,1s",      # C原子の1s軌道
    "0, C,2s",      # C原子の2s軌道
    "0, C,3s",      # C原子の3s軌道
    "0, C,2px",     # C原子の2px軌道
    "0, C,2py",     # C原子の2py軌道
    "0, C,2pz",     # C原子の2pz軌道
    "0, C,3px",     # C原子の3px軌道
    "0, C,3py",     # C原子の3py軌道
    "0, C,3pz",     # C原子の3pz軌道
    "0, C,3dxy",    # C原子の3dxy軌道
    "0, C,3dyz",    # C原子の3dyz軌道
    "0, C,3dz^2",   # C原子の3dz^2軌道
    "0, C,3dxz",    # C原子の3dxz軌道
    "0, C,3dx2-y2", # C原子の3dx2-y2軌道
    "1, H,1s",      # H原子1の1s軌道
    "1, H,2s",      # H原子1の2s軌道
    "1, H,2px",     # H原子1の2px軌道
    "1, H,2py",     # H原子1の2py軌道
    "1, H,2pz",     # H原子1の2pz軌道
    # ... 他のH原子の基底関数
]
```

#### 2. element_to_index辞書
```python
element_to_index = {
    "H": 0,
    "S": 1,  # 特別なハック
    "Li": 2,
    "Be": 3,
    "B": 4,
    "C": 5,
    "N": 6,
    "O": 7,
    "F": 8,
    # ... 他の元素
}
```

### 出力データの例

#### 1. atom_type_embedding_per_atom
```python
# メタン分子（CH4）の場合
atom_type_embedding_per_atom = {
    0: 5,  # C原子 → インデックス5
    1: 0,  # H原子1 → インデックス0
    2: 0,  # H原子2 → インデックス0
    3: 0,  # H原子3 → インデックス0
    4: 0   # H原子4 → インデックス0
}
```

#### 2. irrep_string_per_atom
```python
# メタン分子（CH4）の場合
irrep_string_per_atom = {
    0: "1x0e+1x0e+1x0e+1x1o+1x1o+1x2e",  # C原子
    1: "1x0e+1x0e+1x1o",                  # H原子1
    2: "1x0e+1x0e+1x1o",                  # H原子2
    3: "1x0e+1x0e+1x1o",                  # H原子3
    4: "1x0e+1x0e+1x1o"                   # H原子4
}
```

#### 3. irrep文字列の詳細解析
```python
# C原子のirrep文字列: "1x0e+1x0e+1x0e+1x1o+1x1o+1x2e"
# 各既約表現の意味:
# - 1x0e: s軌道（スカラー、1次元）
# - 1x0e: s軌道（スカラー、1次元）
# - 1x0e: s軌道（スカラー、1次元）
# - 1x1o: p軌道（ベクトル、3次元）
# - 1x1o: p軌道（ベクトル、3次元）
# - 1x2e: d軌道（テンソル、5次元）
# 合計次元: 1 + 1 + 1 + 3 + 3 + 5 = 14次元

# H原子のirrep文字列: "1x0e+1x0e+1x1o"
# 各既約表現の意味:
# - 1x0e: s軌道（スカラー、1次元）
# - 1x0e: s軌道（スカラー、1次元）
# - 1x1o: p軌道（ベクトル、3次元）
# 合計次元: 1 + 1 + 3 = 5次元
```

## 重要な特徴

### 1. 軌道タイプの変換規則
- **s軌道**: `1x0e` (スカラー、回転不変)
- **p軌道**: `1x1o` (ベクトル、回転等変)
- **d軌道**: `1x2e` (テンソル、回転等変)

### 2. 原子種のインデックス化
- 各元素に一意のインデックスを割り当て
- 機械学習での原子種識別に使用
- `element_to_index`辞書で管理

### 3. 基底関数の順序
- PySCFの基底関数順序に従う
- 原子ごとにグループ化
- 軌道角運動量の順序（s, p, d, f...）

### 4. E3NN互換性
- 出力はE3NNの既約表現形式に準拠
- 回転不変性と等変性を保持
- 機械学習モデルで直接使用可能

## エラーハンドリング

### よくあるエラーと対処法

1. **未定義の元素**
   ```python
   # エラー: KeyError for unknown element
   # 対処: element_to_index辞書に元素を追加
   element_to_index["新元素"] = 新しいインデックス
   ```

2. **基底関数の形式エラー**
   ```python
   # エラー: 基底関数ラベルの解析失敗
   # 対処: mol.ao_labels()のフォーマットを確認
   ao_labels = mol.ao_labels(fmt="%d,%2s,%s%-0s")
   ```

3. **分子オブジェクトの未構築**
   ```python
   # エラー: 分子オブジェクトが構築されていない
   # 対処: mol.build()を実行
   mol.build()
   ```

## パフォーマンス考慮事項

### 1. 計算効率
- 基底関数数に比例した計算時間
- 大きな分子では処理時間が増加

### 2. メモリ使用量
- 原子数と基底関数数に比例
- 大きな分子ではメモリ使用量が増加

### 3. 並列処理
- 各原子の処理は独立
- 並列化可能な構造

## テスト結果からの知見

テストスイートから得られた重要な知見：

1. **信頼性**: 様々な分子（H2, CH4, H2O）で安定動作
2. **一貫性**: 同じ分子に対して再現可能な出力
3. **正確性**: 期待されるirrep文字列と原子種インデックスの生成
4. **堅牢性**: 異なる基底関数セットでの動作確認

## 関連関数

- `embed_mos_on_atoms`: 生成されたirrep文字列を使用して分子軌道を埋め込み
- `get_max_n_per_l_for_basis`: 基底関数セットから最大軌道数を決定
- `fix_d_orbitals`: PySCFとE3NNのd軌道順序の違いを修正

## まとめ

`mos_to_irrep_strings`関数は、量子化学計算と機械学習を橋渡しする重要な関数です。PySCFの基底関数情報をE3NNで処理可能な既約表現形式に変換し、原子種の識別に必要なインデックスを生成します。この関数により、分子の構造情報を機械学習モデルで効率的に処理できるようになります。

**主な役割**:
1. **基底関数の解析**: PySCFの基底関数ラベルから軌道情報を抽出
2. **E3NN変換**: 軌道タイプをE3NN既約表現に変換
3. **原子種インデックス化**: 元素を機械学習用のインデックスに変換
4. **構造化出力**: 原子ごとに整理されたirrep文字列を生成

この関数は`embed_mos_on_atoms`関数と連携して、分子軌道データを機械学習で使用可能な形式に変換する重要な役割を果たします。
