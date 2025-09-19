# QM7Maker.py 解説

## 概要

`qm7maker.py`は、QM7データセットをダウンロードし、量子化学計算（CCSD）を実行して分子の電子状態を計算し、ローカライゼーション処理を行うスクリプトです。このスクリプトは、機械学習モデルの訓練用データセットを準備するために使用されます。

## 主な機能

1. **QM7データセットのダウンロード**: 量子機械学習で広く使用されるQM7データセットを自動ダウンロード
2. **分子データの前処理**: 原子座標の単位変換（Bohr → Angstrom）
3. **量子化学計算**: CCSD（Coupled Cluster Singles and Doubles）法による電子状態計算
4. **ローカライゼーション**: 分子軌道のローカライゼーション処理
5. **並列処理**: 複数の分子を並列で処理可能

## 定数と設定

```python
MAX_MEM = 64000  # メモリ使用量制限（MB）
BASIS = 'def2-svp'  # 基底関数セット
QM7_URL = "http://www.quantum-machine.org/data/qm7.mat"  # QM7データセットのURL
```

## コマンドライン引数

### 必須引数
- `--config_profile_path`: 設定ファイルのパス（デフォルト: `configs/profile/my_profile.yaml`）

### オプション引数
- `--length`: 使用するデータセットの長さ（カスタムインデックスが指定されていない場合のみ）
- `--random`: QM7のランダムサブセットを使用
- `--indices_file`: 使用するインデックスのテキストファイルパス（`--random`と排他的）
- `--indices_out`: 使用したインデックスを出力するファイルパス
- `--num_workers`: 並列処理のワーカー数（デフォルト: 1）

## 主要な関数

### `parser_helper()`
コマンドライン引数を解析し、設定を返す関数。

### `_init_worker(z_array, r_array)`
並列処理用のワーカープロセスを初期化する関数。グローバル変数として原子番号配列（`_Z_ARRAY`）と座標配列（`_R_ARRAY`）を設定。

### `_process_molecule_idx(task_tuple)`
個別の分子を処理する関数：
1. 原子番号と座標を抽出
2. 座標をBohrからAngstromに変換（0.529177倍）
3. CCSD計算を実行
4. ローカライゼーション処理を実行

## 処理フロー

### 1. 初期設定
```python
# OpenMPスレッド数を制限（PySCFの過剰使用を防ぐ）
os.environ.setdefault("OMP_NUM_THREADS", "1")
```

### 2. データセットダウンロード
- 設定ファイルからデータセットパスを取得
- QM7データセットが存在しない場合は自動ダウンロード

### 3. データ読み込み
```python
mat_data = loadmat(path_of_mat)
dataset_length = mat_data['Z'].shape[0]  # 分子数
```

### 4. インデックス選択
以下の優先順位で処理：
1. `--indices_file`が指定されている場合：ファイルからインデックスを読み込み
2. `--length`が指定されている場合：
   - `--random`フラグがあればランダム選択
   - なければ最初のN個を使用
3. どちらも指定されていない場合：全データセットを使用

### 5. 分子処理
#### 並列処理（`num_workers > 1`）
```python
with ProcessPoolExecutor(
    max_workers=args.num_workers,
    mp_context=mp_ctx,
    initializer=_init_worker,
    initargs=(mat_data['Z'], mat_data['R'])
) as executor:
    for _ in executor.map(_process_molecule_idx, tasks):
        pass
```

#### 逐次処理（`num_workers = 1`）
各分子に対して順次処理を実行。

## 出力ファイル

### 分子データファイル
- 形式: `mol_{index}.chk`
- 内容: CCSD計算結果とローカライゼーション結果

### インデックスファイル
- ファイル名: `selected_indices.txt`（デフォルト）
- 内容: 処理に使用した分子のインデックス（1行に1つ）

## 使用例

### 基本的な使用方法
```bash
python qm7maker.py --config_profile_path configs/profile/my_profile.yaml
```

### ランダムサブセット（100分子）
```bash
python qm7maker.py --length 100 --random --num_workers 4
```

### カスタムインデックスファイルを使用
```bash
python qm7maker.py --indices_file my_indices.txt --num_workers 8
```

## 注意事項

1. **メモリ使用量**: `MAX_MEM`で制限されているが、大きな分子ではメモリ不足の可能性
2. **並列処理**: `OMP_NUM_THREADS=1`に設定されているため、各ワーカーは単一スレッドで動作
3. **エラーハンドリング**: 並列処理でエラーが発生した場合、自動的に逐次処理にフォールバック
4. **再現性**: `np.random.seed(42)`でランダム選択の再現性を保証

## 依存関係

- `egsmole.utils.chkfile_utils`: CCSD計算とローカライゼーション機能
- `scipy.io.loadmat`: MATLABファイルの読み込み
- `omegaconf.OmegaConf`: 設定ファイルの管理
- `concurrent.futures.ProcessPoolExecutor`: 並列処理

## 出力メッセージ

スクリプト実行時には以下の情報が表示されます：
- ダウンロード状況
- データセットの分子数
- 使用するデータパス
- 処理方法（並列/逐次）
- 完了メッセージと設定の更新指示
