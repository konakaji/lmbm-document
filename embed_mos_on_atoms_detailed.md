# embed_mos_on_atoms 関数詳細ドキュメント

## 概要

`embed_mos_on_atoms`関数は、分子軌道係数（MO coefficients）を各原子に埋め込み、E3NN（Equivariant Neural Networks）で使用可能な形式に変換する重要な関数です。この関数は、量子化学計算で得られた分子軌道データを機械学習モデルで処理可能な構造化されたテンソル形式に変換します。

## 関数シグネチャ

```python
def embed_mos_on_atoms(
    mo_coeff: torch.Tensor,
    max_n_per_l: Dict[int, int],
    irrep_string_per_atom: List[str],
    atom_type_embedding_per_atom: torch.Tensor,
    mo_energies: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, e3nn.o3.Irreps, torch.Tensor, e3nn.o3.Irreps]
```

## パラメータ詳細

### 入力パラメータ

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `mo_coeff` | `torch.Tensor` | 分子軌道係数テンソル。形状: `(n_basis, n_mo)`（PySCF由来） |
| `max_n_per_l` | `Dict[int, int]` | 各軌道角運動量lに対する最大軌道数の辞書（主量子数の最大値に対応） |
| `irrep_string_per_atom` | `List[str]` | 各原子のE3NN既約表現文字列のリスト |
| `atom_type_embedding_per_atom` | `torch.Tensor` | 各原子の原子種埋め込みベクトル |
| `mo_energies` | `Optional[torch.Tensor]` | 分子軌道エネルギー（オプション） |
| `dtype` | `torch.dtype` | 出力テンソルのデータ型（デフォルト: float32） |

### 出力

4つのタプルを返します：

1. `mo_embeddings_padded`: パディングされた分子軌道埋め込み `(n_mo, n_atom, n_features)`
2. `mo_irreps_padded_sorted`: ソートされたE3NN既約表現
3. `mo_embeddings_padded_with_atom_idx`: 原子種情報を含む埋め込み `(n_mo, n_atom, n_features_with_atom_type)`
4. `mo_irreps_padded_sorted_with_atom_idx`: 原子種情報を含むE3NN既約表現

## 処理フロー

### 1. 前処理
- **d軌道修正**: PySCFとE3NNのd軌道順序の違いを`fix_d_orbitals`で修正
- **転置処理**: `mo_coeff`を`(n_basis, n_mo)`から`(n_mo, n_basis)`へ転置

### 2. 各原子の軌道処理
- **既約表現解析**: `e3nn.o3.Irreps`で原子の既約表現を解析
- **不足分計算**: `l_counts`と`target_ls`から`l_diff`（不足分）を計算
- **パディング軌道生成**: `diff_irreps`を生成し、`mo_irreps_padded = mo_irreps + diff_irreps`で結合

### 3. 軌道係数の抽出とパディング
- **係数抽出**: `atom_coefs`を抽出
- **ゼロパディング**: `torch.nn.functional.pad`でゼロパディング

### 4. 原子種エンコーディング
- **one-hot生成**: `atom_type_embedding_per_atom`から原子種のone-hotベクトルを生成
- **特徴結合**: `atom_type_one_hot`と`atom_coeffs_paddded`を結合

### 5. エネルギー統合（オプション）
- **エネルギー追加**: `mo_energies`が提供された場合、`1x0e`既約表現として追加

## 使用例

### 基本的な使用例

```python
import torch
import numpy as np
from pyscf import gto, scf
from egsmole.utils import hf_utils

# 分子の定義（メタン分子）
mol_string = """
    C   0.0  0.0  0.0
    H   1.0  1.0  1.0
    H  -1.0 -1.0  1.0
    H  -1.0  1.0 -1.0
    H   1.0 -1.0 -1.0
"""

# PySCFで分子軌道計算
mol = gto.Mole()
mol.atom = mol_string
mol.basis = "def2-svp"
mol.build()

# Hartree-Fock計算を実行
mf = scf.RHF(mol)
mf.kernel()  # ここで分子軌道係数が計算される

# 分子軌道係数を取得（PySCFのmf.mo_coeffから）
mo_coeff = torch.tensor(mf.mo_coeff, dtype=torch.float32)
# 形状: (n_basis, n_mo) = (34, 34) for CH4

# 必要なパラメータを準備
max_n_per_l = hf_utils.get_max_n_per_l_for_basis("def2-svp")
atom_type_embedding_per_atom, irrep_string_per_atom = hf_utils.mos_to_irrep_strings(mol)

# 分子軌道エネルギー（オプション）
mo_energies = torch.tensor(mf.mo_energy, dtype=torch.float32)

# embed_mos_on_atoms関数を実行
result = hf_utils.embed_mos_on_atoms(
    mo_coeff,
    max_n_per_l,
    irrep_string_per_atom,
    atom_type_embedding_per_atom,
    mo_energies=mo_energies,
    dtype=torch.float32
)

mo_embeddings_padded, mo_irreps_padded_sorted, mo_embeddings_padded_with_atom_idx, mo_irreps_padded_sorted_with_atom_idx = result

print(f"MO embeddings shape: {mo_embeddings_padded.shape}")
print(f"MO irreps: {mo_irreps_padded_sorted}")
print(f"MO embeddings with atom type shape: {mo_embeddings_padded_with_atom_idx.shape}")
```

### 入力データの例

#### 1. 分子軌道係数 (mo_coeff)
```python
# メタン分子（CH4）の例
# 形状: (n_basis, n_mo) = (34, 34)
# PySCFのmf.mo_coeffから取得
mo_coeff = torch.tensor([
    [0.992, -0.257, 0.000, 0.000, 0.000, ...],  # C 1s基底関数
    [0.029, 0.551, 0.000, 0.000, 0.000, ...],   # C 2s基底関数
    [-0.015, 0.363, 0.000, 0.000, 0.000, ...],  # C 3s基底関数
    [0.000, 0.000, 0.295, 0.010, 0.211, ...],   # C 2px基底関数
    [0.000, 0.000, -0.211, -0.004, 0.295, ...], # C 2py基底関数
    # ... 他の基底関数（合計34個）
], dtype=torch.float32)
```

**mo_coeffの物理的意味**:
- **PySCF由来**: Hartree-Fock計算（`mf.kernel()`）で得られる分子軌道係数
- **形状**: `(n_basis, n_mo)` = `(基底関数数, 分子軌道数)`
- **内容**: `mo_coeff[i, j]` = 基底関数iが分子軌道jに寄与する係数
- **各行**: 1つの基底関数が各分子軌道にどの程度寄与するか
- **各列**: 1つの分子軌道が各基底関数からどの程度構成されるか
- **例**: メタン分子では34個の基底関数から34個の分子軌道が構成される

#### 2. max_n_per_l パラメータ
```python
# def2-svp基底関数セットの場合
max_n_per_l = {
    0: 4,  # s軌道: 最大4個 (主量子数 n=1,2,3,4)
    1: 3,  # p軌道: 最大3個 (主量子数 n=2,3,4)
    2: 1   # d軌道: 最大1個 (主量子数 n=3)
}
```

**max_n_per_lの物理的意味**:
- 各軌道角運動量lに対して、**考慮する元素の中で最大の主量子数**を表す
- s軌道が4個なのは、第3周期元素（Na-Ar）が4個のs軌道（1s, 2s, 3s, 4s）を持つため
- 軽い元素（H, C, Oなど）は3個のs軌道しか使わないが、パディングによって4個目はゼロで埋められる
- これにより、異なる元素を含む分子でも統一された次元でバッチ処理が可能

#### 3. irrep_string_per_atom
```python
# メタン分子（CH4）の場合
irrep_string_per_atom = [
    "1x0e+1x1o+1x2e",  # C原子: s, p, d軌道
    "1x0e",            # H原子1: s軌道のみ
    "1x0e",            # H原子2: s軌道のみ
    "1x0e",            # H原子3: s軌道のみ
    "1x0e"             # H原子4: s軌道のみ
]
```

#### 4. atom_type_embedding_per_atom
```python
# 原子種の埋め込みベクトル
# 形状: (n_atom, n_atom_types)
atom_type_embedding_per_atom = torch.tensor([
    [1, 0],  # C原子 (原子種0)
    [0, 1],  # H原子 (原子種1)
    [0, 1],  # H原子 (原子種1)
    [0, 1],  # H原子 (原子種1)
    [0, 1]   # H原子 (原子種1)
], dtype=torch.float32)
```

#### 5. 分子軌道エネルギー (mo_energies)
```python
# 分子軌道エネルギー
# 形状: (n_mo,)
mo_energies = torch.tensor([
    -20.5,  # 1s軌道エネルギー
    -1.2,   # 2s軌道エネルギー
    -0.8,   # 2px軌道エネルギー
    -0.8,   # 2py軌道エネルギー
    -0.8,   # 2pz軌道エネルギー
    0.3,    # 3s軌道エネルギー
    0.5,    # 3px軌道エネルギー
    0.5,    # 3py軌道エネルギー
    0.5,    # 3pz軌道エネルギー
    1.2     # 3d軌道エネルギー
], dtype=torch.float32)
```

### 出力データの例

#### 1. mo_embeddings_padded
```python
# 形状: (n_mo, n_atom, n_features) = (10, 5, 19)
# 各分子軌道が各原子にどのように寄与するかを表現
mo_embeddings_padded = torch.tensor([
    # 1s軌道の各原子への寄与
    [[0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.5],  # C原子
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.5],  # H原子1
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.5],  # H原子2
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.5],  # H原子3
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.5]], # H原子4
    
    # 2s軌道の各原子への寄与
    [[0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.2],   # C原子
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.2],   # H原子1
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.2],   # H原子2
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.2],   # H原子3
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.2]],  # H原子4
    
    # ... 他の分子軌道
], dtype=torch.float32)
```

#### 2. mo_irreps_padded_sorted
```python
# E3NN既約表現（エネルギー含む）
mo_irreps_padded_sorted = e3nn.o3.Irreps("1x0e+1x1o+1x2e+1x0e")
# 説明:
# - 1x0e: s軌道（スカラー）
# - 1x1o: p軌道（ベクトル）
# - 1x2e: d軌道（テンソル）
# - 1x0e: エネルギー（スカラー）
```

#### 3. mo_embeddings_padded_with_atom_idx
```python
# 形状: (n_mo, n_atom, n_features_with_atom_type) = (10, 5, 21)
# 原子種情報を含む埋め込み（2次元追加）
mo_embeddings_padded_with_atom_idx = torch.tensor([
    # 1s軌道の各原子への寄与（原子種情報含む）
    [[0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.5, 1, 0],  # C原子
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.5, 0, 1],  # H原子1
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.5, 0, 1],  # H原子2
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.5, 0, 1],  # H原子3
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.5, 0, 1]], # H原子4
    
    # ... 他の分子軌道
], dtype=torch.float32)
```

## 重要な特徴

### 1. パディング処理
- 異なる原子間で軌道数の違いを統一するため、ゼロパディングを適用
- バッチ処理での効率的な計算を可能にする
- **主量子数の違いによるパディング**: 軽い元素は少ない主量子数しか使わないが、重い元素の最大主量子数に合わせてパディング

### 2. E3NN互換性
- 出力はE3NNの既約表現形式に準拠
- 回転不変性と等変性を保持

### 3. 原子種情報の統合
- 原子種のone-hotエンコーディングを特徴ベクトルに統合
- 機械学習モデルが原子種を識別可能

### 4. エネルギー情報の統合
- 分子軌道エネルギーを`1x0e`既約表現として追加
- 軌道のエネルギー準位情報を保持

## エラーハンドリング

### よくあるエラーと対処法

1. **テンソル形状の不一致**
   ```python
   # エラー: mo_coeffの形状が(n_mo, n_basis)の場合
   # 対処: 転置して(n_basis, n_mo)にする
   mo_coeff = mo_coeff.T
   ```

2. **原子数とirrep文字列数の不一致**
   ```python
   # エラー: len(irrep_string_per_atom) != n_atom
   # 対処: 分子の原子数と一致することを確認
   assert len(irrep_string_per_atom) == mol.natm
   ```

3. **データ型の不一致**
   ```python
   # エラー: 異なるデータ型のテンソル
   # 対処: 統一されたデータ型を使用
   mo_coeff = mo_coeff.to(dtype=torch.float32)
   ```

## パフォーマンス考慮事項

### 1. メモリ使用量
- 大きな分子では出力テンソルが大きくなる
- 必要に応じてバッチサイズを調整

### 2. 計算効率
- パディング処理は計算オーバーヘッドを伴う
- 事前に`max_n_per_l`を適切に設定することで効率化

### 3. 並列処理
- 複数の分子を並列処理する場合は、各分子の原子数が異なることに注意

## テスト結果からの知見

テストスイート（12個のテスト）から得られた重要な知見：

1. **信頼性**: 様々な分子（H2, CH4, H2O）で安定動作
2. **一貫性**: 同じ入力に対して再現可能な出力
3. **堅牢性**: エッジケースに対する適切なエラーハンドリング
4. **正確性**: 期待される形状とデータ型での出力

## 関連関数

- `calculate_hf_mos`: PySCFでHartree-Fock計算を実行し、mo_coeffを取得
  - `mf.kernel()`で分子軌道係数を計算
  - `mf.mo_coeff`から分子軌道係数を取得（形状: (n_basis, n_mo)）
- `get_max_n_per_l_for_basis`: 基底関数セットからmax_n_per_lを自動決定
  - 全元素をスキャンして各軌道角運動量lに対する最大の主量子数を決定
  - 例：def2-svpではs軌道4個（Na-Ar元素が4個のs軌道を持つため）
- `mos_to_irrep_strings`: 分子からirrep文字列を生成
- `fix_d_orbitals`: PySCFとE3NNのd軌道順序の違いを修正

## まとめ

`embed_mos_on_atoms`関数は、量子化学計算と機械学習を橋渡しする重要な関数です。分子軌道データをE3NNで処理可能な形式に変換し、回転不変性と等変性を保持しながら、原子種情報とエネルギー情報を統合した構造化された表現を提供します。

**max_n_per_lの本質**は、考慮する元素の中で最大の主量子数を表しており、これにより異なる元素を含む分子でも統一された次元でバッチ処理が可能になります。例えば、def2-svp基底関数セットでは、第3周期元素（Na-Ar）が4個のs軌道（1s, 2s, 3s, 4s）を持つため、s軌道の最大数は4個となり、軽い元素はパディングによって統一された次元で処理されます。
