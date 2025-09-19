# DataPrep.py 解説

## 概要

`dataprep.py`は、`qm7maker.py`で生成された量子化学計算結果（`.chk`ファイル）を読み込み、機械学習モデルの訓練に適した形式に前処理するスクリプトです。このスクリプトは、Hydraフレームワークを使用して設定管理を行い、複雑なデータ前処理パイプラインを実行します。

## 主な機能

1. **量子化学データの読み込み**: PySCFの`.chk`ファイルから分子軌道、CCSD振幅、MP2振幅を読み込み
2. **分子軌道の変換**: 正準軌道から局在化軌道への変換、直交化処理
3. **データの正規化**: 分子軌道埋め込みの正規化とパディング
4. **グラフ構造の生成**: 分子の幾何学的構造をグラフ形式に変換
5. **並列処理**: 複数の分子を並列で前処理
6. **キャッシュ機能**: 前処理済みデータの保存と再利用

## アーキテクチャ

### メインエントリーポイント
```python
@hydra.main(version_base="1.3.2", config_path="configs", config_name="default")
def main(config: DictConfig):
    dataprep = hydra.utils.instantiate(config.dataset.dataprep)
    dataprep.run()
```

### 主要クラス

#### 1. MODataprep
データ前処理のメインクラス。以下の処理を実行：

- **データ読み込み**: `.chk`ファイルから量子化学計算結果を読み込み
- **前処理**: 分子軌道の変換、正規化、パディング
- **保存**: 前処理済みデータを`.pt`ファイルとして保存

#### 2. DataHolder
個々の分子データを格納するデータクラス：

```python
@dataclass
class DataHolder:
    idx: int                           # 分子インデックス
    file: str                          # ファイルパス
    atom_type: list                    # 原子種
    n_core_electrons: torch.Tensor     # コア電子数
    atom_positions: list               # 原子座標
    mo_energies: torch.Tensor          # 分子軌道エネルギー
    e_corr_mp2: float                  # MP2相関エネルギー
    e_corr_ccsd: float                 # CCSD相関エネルギー
    t1_ccsd_canonical: torch.Tensor    # CCSD T1振幅（正準）
    t1_ccsd_localized: torch.Tensor    # CCSD T1振幅（局在化）
    t2_ccsd_canonical: torch.Tensor    # CCSD T2振幅（正準）
    t2_ccsd_localized: torch.Tensor    # CCSD T2振幅（局在化）
    # ... その他のフィールド
```

## データ処理パイプライン

### 1. ファイル読み込み (`load_file_impl`)

```python
def load_file_impl(file, dtype, compute_localized=True, compute_orthogonal=True, load_integrals=False):
```

**処理内容:**
- PySCFの`.chk`ファイルから分子情報を読み込み
- HDF5ファイルから量子化学計算結果を読み込み：
  - 分子軌道係数 (`mo_coeff`)
  - 分子軌道エネルギー (`mo_energy`)
  - MP2相関エネルギー (`e_corr_mp2`)
  - CCSD相関エネルギー (`e_corr_ccsd`)
  - CCSD振幅 (`t1`, `t2`)
  - MP2振幅 (`t2_mp2`)

**座標変換:**
- 原子座標をAngstrom単位で取得
- 重なり積分行列の平方根を計算（直交化用）

### 2. 分子軌道の変換

#### 正準軌道 → 局在化軌道
```python
if compute_localized:
    t1_ccsd_loc = transform_amplitudes(t1_ccsd, mo_coeff, mo_coeff_loc)
    t2_ccsd_loc = transform_amplitudes(t2_ccsd, mo_coeff, mo_coeff_loc)
    t2_mp2_loc = transform_amplitudes(t2_mp2, mo_coeff, mo_coeff_loc)
```

#### 直交化処理
```python
s_half = np.real(sqrtm(ovlp))  # 重なり積分行列の平方根
mo_orthog = s_half @ mo_coeff  # 直交化された分子軌道
```

### 3. 分子軌道の埋め込み (`process_single_file_data_impl`)

```python
def process_single_file_data_impl(dh: DataHolder, basis_set, max_n_per_l, max_n_oc, max_n_vi, max_n_atom, do_padding: bool, compute_integral: bool, dtype: torch.dtype):
```

**処理内容:**
- 原子種から分子オブジェクトを生成
- 分子軌道を原子上に埋め込み
- パディング処理（分子間のサイズ統一）
- フローズンコアエネルギーの計算

### 4. グラフ構造の生成

分子の幾何学的構造をグラフ形式に変換：
- **ノード**: 原子（位置、原子種）
- **エッジ**: 原子間の結合（距離、ベクトル）
- **特徴**: 分子軌道埋め込み

## 設定パラメータ

### 基本設定
```yaml
dataprep:
  _target_: egsmole.dataset.MODataprep.MODataprep
  dataset_path: ${dataset.dataset_path}      # 出力パス
  dataset_inpath: ${dataset.dataset_inpath}  # 入力パス（.chkファイル）
  basis_set: ${model.basis_set}              # 基底関数セット
  data_type: ccsd                            # データタイプ
  dtype: float32                             # データ型
```

### 処理オプション
```yaml
  compute_localized: true      # 局在化軌道の計算
  compute_orthogonal: true     # 直交化処理
  compute_integral: true       # 積分の計算
  mo_type: "localized_orthogonal"  # 使用する分子軌道タイプ
```

### 並列処理設定
```yaml
  num_workers: 16             # 並列ワーカー数
  n_process_chunk_size: 100   # チャンクサイズ
```

## 分子軌道タイプ

4つの分子軌道タイプが利用可能：

1. **`canonical_original`**: 正準軌道（元の形式）
2. **`canonical_orthogonal`**: 正準軌道（直交化済み）
3. **`localized_original`**: 局在化軌道（元の形式）
4. **`localized_orthogonal`**: 局在化軌道（直交化済み）

## 出力データ構造

### 前処理済みファイル
- **場所**: `{dataset_path}/preprocessed/processed_data/`
- **形式**: `.pt`ファイル（PyTorch形式）
- **内容**: 各分子の前処理済みデータ

### メタデータ
- **統計情報**: 分子数、原子数、軌道数の統計
- **正規化パラメータ**: エネルギーの平均・標準偏差
- **インデックスファイル**: 処理された分子のリスト

## 使用例

### 基本的な使用方法
```bash
python dataprep.py
```

### 設定ファイルの指定
```bash
python dataprep.py --config-name=qm7_100_random
```

### カスタム設定のオーバーライド
```bash
python dataprep.py dataprep.num_workers=32 dataprep.mo_type=canonical_original
```

## パフォーマンス最適化

### 並列処理
- **マルチプロセシング**: 複数の分子を並列処理
- **チャンク処理**: メモリ使用量を制御
- **ワーカー数**: CPUコア数に応じて調整

### メモリ管理
- **チャンクサイズ**: 一度に処理する分子数を制御
- **データ型**: `float32`でメモリ使用量を削減
- **キャッシュ**: 前処理済みデータの再利用

## エラーハンドリング

### ファイル読み込みエラー
- 破損した`.chk`ファイルのスキップ
- エラーログの記録

### メモリ不足
- チャンクサイズの自動調整
- 並列ワーカー数の制限

## 依存関係

### 主要ライブラリ
- **Hydra**: 設定管理
- **PyTorch**: テンソル操作
- **PySCF**: 量子化学計算
- **H5Py**: HDF5ファイル読み込み
- **SciPy**: 数値計算

### 内部モジュール
- **`egsmole.utils.hf_utils`**: 分子軌道処理
- **`egsmole.utils.data_utils`**: データ処理ユーティリティ
- **`egsmole.utils.chkfile_utils`**: チェックポイントファイル処理

## トラブルシューティング

### よくある問題

1. **メモリ不足**
   - `n_process_chunk_size`を小さくする
   - `num_workers`を減らす

2. **ファイル読み込みエラー**
   - `.chk`ファイルの整合性を確認
   - ファイルパスの設定を確認

3. **並列処理エラー**
   - `num_workers=1`で逐次処理にフォールバック
   - システムリソースを確認

## MO Embedding変換の詳細

### 1. 分子軌道係数の変換プロセス

`dataprep.py`では、元々の分子軌道係数（MO coefficients）を複数の段階で変換しています：

#### 段階1: 正準軌道から局在化軌道への変換
```python
# 正準軌道係数 (mo_coeff) から局在化軌道係数 (mo_coeff_loc) への変換
mo_coeff_loc = f5["scf/mo_coeff_loc"][:]  # qm7maker.pyで事前計算済み
```

#### 段階2: 直交化処理
```python
# 重なり積分行列の平方根を計算
ovlp = mol.intor_symmetric("int1e_ovlp")
s_half = np.real(sqrtm(ovlp))

# 直交化された分子軌道係数を生成
mo_orthog = s_half @ mo_coeff                    # 正準軌道の直交化
mo_loc_orthog = s_half @ mo_coeff_loc            # 局在化軌道の直交化
```

#### 段階3: 原子上への埋め込み変換
```python
def embed_mos_on_atoms(mo_coeff, max_n_per_l, irrep_string_per_atom, atom_type_embedding_per_atom):
    # 各原子の軌道をE3NNの既約表現形式に変換
    for atom_idx, atom_irrep in irrep_string_per_atom.items():
        mo_irreps = e3nn.o3.Irreps(atom_irrep)
        # パディング処理で軌道数を統一
        # 原子種のone-hotエンコーディングを追加
```

### 2. 4つのMO Embeddingタイプ

#### `canonical_original`
- **元データ**: 正準軌道係数（`mo_coeff`）
- **特徴**: 分子全体に広がった軌道
- **用途**: 基本的な分子軌道表現

#### `canonical_orthogonal`
- **元データ**: 正準軌道係数（`mo_coeff`）
- **変換**: `s_half @ mo_coeff`
- **特徴**: 直交化された正準軌道
- **用途**: 数値的安定性を向上

#### `localized_original`
- **元データ**: 局在化軌道係数（`mo_coeff_loc`）
- **特徴**: 特定の原子や結合に局在した軌道
- **用途**: 化学的解釈が容易

#### `localized_orthogonal`
- **元データ**: 局在化軌道係数（`mo_coeff_loc`）
- **変換**: `s_half @ mo_coeff_loc`
- **特徴**: 直交化された局在化軌道
- **用途**: 機械学習に最適化された表現

### 3. 軌道変換の数学的詳細

#### 直交化変換
```python
# 重なり積分行列 S の平方根を計算
S = mol.intor_symmetric("int1e_ovlp")  # 重なり積分行列
S_half = sqrtm(S)                      # S^(1/2)

# 直交化変換
C_orthog = S_half @ C_original
```

この変換により、以下の条件を満たす軌道係数が得られます：
- `C_orthog^T @ C_orthog = I` (直交性)
- 元の軌道の物理的意味を保持

#### 局在化変換
局在化軌道は`qm7maker.py`の`localize_dump`関数で事前計算されます：

```python
def localize(mol, mo_coeff, fock, localize_config):
    # Boys局在化法を使用
    # 軌道の空間的広がりを最小化
    # 化学結合に対応した局在化軌道を生成
```

### 4. E3NN既約表現への変換

#### 軌道タイプの分類
```python
def mos_to_irrep_strings(mol):
    # s軌道: "1x0e" (l=0, パリティ=偶)
    # p軌道: "1x1o" (l=1, パリティ=奇)  
    # d軌道: "1x2e" (l=2, パリティ=偶)
```

#### パディング処理
```python
# 各原子の軌道数を統一
target_ls = [6, 3, 1, 0]  # s, p, d, f軌道の最大数
l_diff = target_ls - l_counts  # 不足分を計算
# ゼロパディングで軌道数を統一
```

### 5. データ構造の変換

#### 入力形式
- **形状**: `(n_basis, n_mo)` - 基底関数×分子軌道
- **内容**: 分子軌道係数行列

#### 出力形式
- **形状**: `(n_mo, n_atom, n_features)` - 分子軌道×原子×特徴
- **内容**: 各原子上の軌道埋め込み

#### 特徴ベクトルの構成
```python
# 各原子の特徴ベクトル
atom_features = [
    atom_type_one_hot,    # 原子種のone-hotエンコーディング
    mo_coefficients,      # 分子軌道係数
    mo_energies          # 分子軌道エネルギー（オプション）
]
```

### 6. パフォーマンス最適化

#### 並列処理
```python
# 複数の分子軌道タイプを並列処理
mo_embeddings_padded = hf_utils.embed_mos_on_atoms(
    dh.mo_embeddings_canonical_original, max_n_per_l, 
    irrep_string_per_atom, atom_type_embedding_per_atom
)
```

#### メモリ効率
- **チャンク処理**: 大きな分子を小さなブロックに分割
- **データ型**: `float32`でメモリ使用量を削減
- **キャッシュ**: 変換済みデータの再利用

### 7. `embed_mos_on_atoms`関数の詳細解析

#### 関数の目的
`embed_mos_on_atoms`関数は、分子軌道係数を各原子に埋め込み、E3NN（Equivariant Neural Networks）で使用可能な形式に変換する重要な関数です。

#### 入力パラメータ
```python
def embed_mos_on_atoms(
    mo_coeff,                           # 分子軌道係数 (n_basis, n_mo)
    max_n_per_l,                        # 各軌道角運動量の最大数 {0: 6, 1: 3, 2: 1, 3: 0}
    irrep_string_per_atom,              # 各原子の既約表現文字列
    atom_type_embedding_per_atom,       # 各原子の原子種インデックス
    mo_energies=None,                   # 分子軌道エネルギー（オプション）
    dtype=torch.float32,                # データ型
):
```

#### 処理ステップの詳細

##### ステップ1: d軌道の修正
```python
# d軌道の順序を修正（PySCFの標準順序からE3NNの期待順序へ）
fix_D = fix_d_orbitals(
    atom_type_embedding_per_atom, irrep_string_per_atom, nbas=mo_coeff.shape[0]
).to(dtype)
mo_coeff = fix_D.T @ mo_coeff
```

**d軌道の順序問題**:
- PySCF: `[d_xx, d_yy, d_zz, d_xy, d_xz, d_yz]`
- E3NN: `[d_xy, d_xz, d_yz, d_xx-yy, d_zz]` (球面調和関数順序)

##### ステップ2: 転置処理
```python
mo_coeff = mo_coeff.T  # (n_basis, n_mo) → (n_mo, n_basis)
```

##### ステップ3: 各原子の軌道処理
```python
for atom_idx, atom_irrep in irrep_string_per_atom.items():
    # 原子の既約表現を解析
    mo_irreps = e3nn.o3.Irreps(atom_irrep)
    
    # 各軌道角運動量の数をカウント
    l_counts = np.array([mo_irreps.ls.count(l) for l in range(len(target_ls))])
    
    # 不足分を計算
    l_diff = target_ls - l_counts
```

**例**: 炭素原子（def2-svp基底）
- `atom_irrep = "1x0e+3x1o+1x2e"` (1つのs軌道、3つのp軌道、1つのd軌道)
- `l_counts = [1, 3, 1, 0]` (s, p, d, f軌道の数)
- `target_ls = [6, 3, 1, 0]` (最大軌道数)
- `l_diff = [5, 0, 0, 0]` (s軌道を5つ追加でパディング)

##### ステップ4: パディング軌道の生成
```python
# 不足分の軌道を生成
diff_irreps = e3nn.o3.Irreps(
    ("".join([f"{m}x{l}{int_to_p[(-1) ** l]}+" for l, m in enumerate(l_diff)]))[:-1]
).simplify()

# 元の軌道とパディング軌道を結合
mo_irreps_padded = mo_irreps + diff_irreps
```

**パディング例**:
- 炭素原子: `"1x0e+3x1o+1x2e" + "5x0e" = "6x0e+3x1o+1x2e"`

##### ステップ5: 軌道係数の抽出とパディング
```python
# 該当原子の軌道係数を抽出
atom_coefs = mo_coeff[:, slice_start : slice_start + mo_irreps.dim]

# ゼロパディング
atom_coeffs_paddded = torch.nn.functional.pad(
    torch.tensor(atom_coefs), (0, diff_irreps.dim), value=0
)
```

##### ステップ6: 原子種のone-hotエンコーディング
```python
atom_type_one_hot = torch.repeat_interleave(
    torch.nn.functional.one_hot(
        torch.tensor(atom_type_embedding_per_atom[atom_idx]),
        num_classes=max_atoms,
    )[None],
    atom_coefs.shape[0],
    dim=0,
)
```

**例**: 炭素原子（インデックス5）
- `atom_type_one_hot = [0, 0, 0, 0, 0, 1, 0, 0, ...]` (9次元、5番目が1)

##### ステップ7: 特徴ベクトルの結合
```python
# 原子種情報と軌道係数を結合
atom_coeffs_paddded_with_atom_idx = torch.cat(
    [atom_type_one_hot, atom_coeffs_paddded], dim=-1
)
```

#### 出力データ構造

##### 基本出力
```python
return (
    mo_embeddings_padded,                    # (n_mo, n_atom, n_features)
    mo_irreps_padded_sorted,                 # 既約表現情報
    mo_embeddings_padded_with_atom_idx,      # 原子種情報付き
    mo_irreps_padded_sorted_with_atom_idx,   # 原子種情報付き既約表現
)
```

##### データ形状の例
```python
# 入力: mo_coeff (n_basis=47, n_mo=23) for C2H4 molecule
# 出力: mo_embeddings_padded (n_mo=23, n_atom=6, n_features=15)

# 各原子の特徴ベクトル構成:
# - 原子種one-hot: 9次元 (H, S, Li, Be, B, C, N, O, F)
# - 軌道係数: 6次元 (パディング済みs軌道)
# 合計: 15次元
```

#### `max_n_per_l`パラメータの決定

##### 基底関数セット別の設定
```python
def get_max_n_per_l_for_basis(basis_set_name="def2-svp", up_to_element="S"):
    # 各元素の基底関数を調べて最大軌道数を決定
    max_n_per_l = {}
    for element in elements:
        basis = gto.load(basis_set_name, element)
        for shell in basis:
            l = shell[0]  # 軌道角運動量量子数
            n_per_l[l] = n_per_l.get(l, 0) + 1
        for l, n in n_per_l.items():
            max_n_per_l[l] = max(max_n_per_l.get(l, 0), n)
    return max_n_per_l
```

**def2-svp基底での例**:
- `max_n_per_l = {0: 6, 1: 3, 2: 1, 3: 0}`
- s軌道: 最大6個、p軌道: 最大3個、d軌道: 最大1個、f軌道: なし

#### 化学的意味

##### 軌道の局在化
- **正準軌道**: 分子全体に広がった軌道
- **局在化軌道**: 特定の原子や結合に局在した軌道

##### パディングの意味
- **物理的意味**: 存在しない軌道はゼロで埋める
- **計算効率**: 全ての分子で同じ次元のテンソルを維持
- **機械学習**: バッチ処理が可能

#### 数値例

##### メタン分子（CH4）の場合
```python
# 入力
mo_coeff.shape = (23, 8)  # 23個の基底関数、8個の分子軌道
atom_type_embedding_per_atom = {0: 5, 1: 0, 2: 0, 3: 0, 4: 0}  # C, H, H, H, H

# 処理後
mo_embeddings_padded.shape = (8, 5, 15)  # 8個の軌道、5個の原子、15次元特徴
```

### 8. 化学的意味

#### 正準軌道 vs 局在化軌道
- **正準軌道**: エネルギー順に並んだ軌道、分子全体に広がる
- **局在化軌道**: 化学結合や孤立電子対に対応、局所的な性質

#### 直交化の重要性
- **数値安定性**: 機械学習アルゴリズムの収束性向上
- **物理的意味**: 軌道の直交性を保持
- **計算効率**: 内積計算の簡素化

## 次のステップ

`dataprep.py`で前処理されたデータは、`MODataset`クラスによって機械学習モデルの训练に使用されます。前処理済みデータは以下の形式で保存され、効率的な训练データローダーとして機能します。

## `diff_irreps`の詳細解析

### 概要
`diff_irreps`は、`embed_mos_on_atoms`関数内で各原子の軌道を統一された形状にするために使用される**パディング用の既約表現**です。異なる原子種でも同じ形状のテンソルを作成し、E3NNでの処理を可能にします。

### 基本パラメータ
```python
# 基底関数セット（def2-svp）から決定される最大軌道数
max_n_per_l = {0: 4, 1: 3, 2: 1}  # s軌道4個、p軌道3個、d軌道1個
target_ls = [4, 3, 1]  # max_n_per_lから生成される目標軌道数
```

### 計算プロセス

#### ステップ1: 現在の軌道数をカウント
```python
l_counts = np.array([mo_irreps.ls.count(l) for l in range(len(target_ls))])
```
- 各原子の現在の軌道数をカウント
- 例：`l_counts = [3, 2, 1]` (s軌道3個、p軌道2個、d軌道1個)

#### ステップ2: 不足分を計算
```python
l_diff = target_ls - l_counts
```
- `target_ls`は`max_n_per_l`から決まる目標軌道数
- 例：`target_ls = [4, 3, 1]`、`l_counts = [3, 2, 1]`
- 結果：`l_diff = [1, 1, 0]` (s軌道1個、p軌道1個不足)

#### ステップ3: パディング用既約表現を生成
```python
diff_irreps = e3nn.o3.Irreps(
    ("".join([f"{m}x{l}{int_to_p[(-1) ** l]}+" for l, m in enumerate(l_diff)]))[:-1]
).simplify()
```

### 原子タイプ別の動作例

#### C原子（原子0, 1, 2）
- **現在の軌道**: `1x0e+1x0e+1x0e+1x1o+1x1o+1x2e` = `3x0e+2x1o+1x2e`
- **`l_counts`**: `[3, 2, 1]` (s軌道3個、p軌道2個、d軌道1個)
- **`l_diff`**: `[1, 1, 0]` (s軌道1個、p軌道1個不足)
- **`diff_irreps`**: `1x0e+1x1o` (不足分を補完)

#### H原子（原子3以降）
- **現在の軌道**: `1x0e+1x0e+1x1o` = `2x0e+1x1o`
- **`l_counts`**: `[2, 1, 0]` (s軌道2個、p軌道1個、d軌道0個)
- **`l_diff`**: `[2, 2, 1]` (s軌道2個、p軌道2個、d軌道1個不足)
- **`diff_irreps`**: `2x0e+2x1o+1x2e` (不足分を補完)

### 文字列生成の詳細
```python
"".join([f"{m}x{l}{int_to_p[(-1) ** l]}+" for l, m in enumerate(l_diff)])
```

- `l=0` (s軌道): `(-1)^0 = 1` → `int_to_p[1] = "e"` → `"1x0e+"`
- `l=1` (p軌道): `(-1)^1 = -1` → `int_to_p[-1] = "o"` → `"1x1o+"`
- `l=2` (d軌道): `(-1)^2 = 1` → `int_to_p[1] = "e"` → `"1x2e+"`

### 最終的な結合
```python
mo_irreps_padded = mo_irreps + diff_irreps
```

**例:**
- **元の軌道**: `3x0e+2x1o+1x2e`
- **パディング**: `1x0e+1x1o`
- **結果**: `4x0e+3x1o+1x2e`

### パディング後の統一形状
すべての原子で最終的に：
- **`mo_irreps_padded_sorted`**: `4x0e+3x1o+1x2e`
- **次元数**: 4×1 + 3×3 + 1×5 = 4 + 9 + 5 = **18次元**

### なぜパディングが必要？

1. **統一性**: すべての原子で同じ形状のテンソルを作る
2. **バッチ処理**: 異なる原子数の分子を同時に処理できる
3. **E3NN互換性**: E3NNが期待する固定形状を提供

### パディングの物理的意味

- **ゼロパディング**: 不足している軌道は係数0で埋める
- **物理的解釈**: その原子に存在しない軌道タイプを表現
- **機械学習**: モデルが軌道の有無を学習できる

## 18次元の構成（最終確認）

### 軌道寄与だけで18次元
- **`mo_irreps_padded_sorted: 4x0e+3x1o+1x2e`の実際の次元計算**
  - **4x0e**: 4 × 1 = 4次元
  - **3x1o**: 3 × 3 = 9次元  
  - **1x2e**: 1 × 5 = 5次元
  - **合計**: 4 + 9 + 5 = **18次元**

### 重要な発見
- **軌道寄与**: **18次元**（`4x0e+3x1o+1x2e`の実際の次元）
- **追加のパディング**: なし（軌道寄与だけで18次元を達成）

### `diff_irreps`の役割（修正版）
`diff_irreps`は各原子の軌道を`4x0e+3x1o+1x2e`の**18次元**に統一するためのパディング軌道を提供している。

- **C原子**: `3x0e+2x1o+1x2e` → `1x0e+1x1o`を追加 → `4x0e+3x1o+1x2e` (18次元)
- **H原子**: `2x0e+1x1o` → `2x0e+2x1o+1x2e`を追加 → `4x0e+3x1o+1x2e` (18次元)

## `torch.cat`と`swapaxes`の処理詳細

### 処理の目的
```python
mo_embeddings_padded = torch.cat(mo_embeddings_padded, dim=0).swapaxes(0, 1)
```

この処理は、各原子の埋め込みデータを1つのテンソルに結合し、E3NNで処理しやすい形状に変換します。

### 処理前の状態
- `mo_embeddings_padded`は**リスト**です
- 各要素は`atom_coeffs_paddded[None]`の形で、各原子の埋め込みデータ
- 例：`[tensor([分子軌道数, 18]), tensor([分子軌道数, 18]), ...]`

### `torch.cat(mo_embeddings_padded, dim=0)`の処理
- **目的**: リスト内のテンソルを結合して1つのテンソルにする
- **結果**: `[分子軌道数, 原子数×18]`の形状
- **例**: 5原子の分子なら`[分子軌道数, 5×18] = [分子軌道数, 90]`

### `.swapaxes(0, 1)`の処理
- **目的**: 第0次元と第1次元を入れ替える
- **結果**: `[原子数×18, 分子軌道数]`の形状
- **例**: `[90, 分子軌道数]`

### 最終的な形状変換

**変換前（リスト）:**
```
[
  tensor([分子軌道数, 18]),  # 原子0のデータ
  tensor([分子軌道数, 18]),  # 原子1のデータ
  tensor([分子軌道数, 18]),  # 原子2のデータ
  ...
]
```

**変換後（テンソル）:**
```
tensor([原子数×18, 分子軌道数])
```

### なぜこの変換が必要？

1. **データ構造の統一**: リストからテンソルへの変換
2. **次元の整理**: 原子ごとのデータを1次元に展開
3. **E3NN互換性**: E3NNが期待する形状への変換

### 具体例（5原子の分子）

```python
# 変換前
mo_embeddings_padded = [
    tensor([34, 18]),  # 原子0
    tensor([34, 18]),  # 原子1  
    tensor([34, 18]),  # 原子2
    tensor([34, 18]),  # 原子3
    tensor([34, 18])   # 原子4
]

# torch.cat後
tensor([34, 90])  # [分子軌道数, 原子数×18]

# swapaxes後  
tensor([90, 34])  # [原子数×18, 分子軌道数]
```

この処理により、各原子の埋め込みデータが1次元に展開され、E3NNで処理しやすい形状になります。
