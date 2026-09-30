# 研究実験テンプレート

DDSPMM と同じ構成を模した Hydra ベースの実験テンプレートです。MNIST の画像識別を最小の動作例として含みます。

## 構成

```
exptemplete/                 # インストール可能な Python パッケージ
  amp/                       # torch.amp の薄いラッパ（autocast / GradScaler）
  bin/                       # CLI（exptemplete-parse-run-command など）
  configs/                   # Hydra YAML（対象ごとに分割）
    dataloader/
    train/
    model/
    optimizer/
    loss/
  loss/
  models/
  utils/
    driver/                  # Trainer / Evaluator
    data/
    wandb/
    _tensorboard/
    _hydra/
recipes/
  _common/local/             # 学習・評価の Python エントリポイント
  MNIST/                     # データセットごとの実行スクリプト
```

Hydra は `exptemplete/configs/config.yaml` をルートに、`dataloader` / `train` / `model` / `optimizer` / `loss` をそれぞれ YAML で切り替え、`hydra.utils.instantiate` でオブジェクト化します。

## セットアップ

Python 3.10 以上を想定しています。

```bash
git clone https://github.com/takemotooo-eeic/exp-templete.git
cd exp-templete
uv sync
# または
pip install -e .
```

パッケージとして import できます。

```python
import exptemplete
from exptemplete.amp import autocast, should_enable_amp
from exptemplete.amp.grad_scaler import GradScaler
```

インストール後は `exptemplete-parse-run-command` がそのまま使えます。

## MNIST の学習

```bash
cd recipes/MNIST
./train.sh
```

主なオプション（`parse_options.sh` 経由）:

```bash
./train.sh --tag cnn_mnist --wandb_enabled true --epochs 10 --data_dir /path/to/data
```

Hydra のグループ切り替えはスクリプト内の変数で行います。

- `dataloader=mnist`
- `train=classification`
- `model=cnn`
- `optimizer=adam`
- `loss=cross_entropy`

チェックポイントは `exp/<tag>/exp/model/` に、TensorBoard ログは `exp/<tag>/tensorboard/` に保存されます。

```bash
tensorboard --logdir exp/<tag>/tensorboard
```

wandb を使う場合は `./train.sh --wandb_enabled true` とし、事前に `wandb login` してください。

## MNIST の評価

```bash
./eval.sh --checkpoint exp/<tag>/exp/model/best_epoch.pth
```

## 新しいデータセットを足す手順

1. `exptemplete/utils/data/` に Dataset を追加する
2. `exptemplete/configs/dataloader/<name>.yaml` を書く
3. 必要なら `model` / `loss` / `train` の YAML も追加する
4. `recipes/<DATASET>/` に `train.sh` / `eval.sh` を置き、`local` を `_common/local` へ symlink する

## AMP

学習ループは `exptemplete.amp` を使っています。半精度にする場合:

```bash
./train.sh train.torch_dtype=float16
```

`float16` / `bfloat16` のときだけ autocast と GradScaler が有効になります。
