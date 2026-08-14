# 🎛️ Remaster Studio

高音質 重低音重視 リマスターツール with AI自動最適化 + Numba JIT高速化

## 機能

- **18種のプロマスタリングプリセット** (Hip-Hop, EDM, Pop, Rock, Jazz等)
- **AI自動最適化** — ジャンル判定 + 最適設定の自動提案
- **バッチ処理** — 複数ファイルの一括リマスター + ZIP一括ダウンロード
- **Numba JIT高速化** — 全DSPループをネイティブコード化（7.4x高速化）
- **パラメトリックEQ** — Sub-Bass, Bass, Mid, High + カスタムバンド
- **マルチバンドコンプレッサー** — 3バンド (Low/Mid/High)
- **Phase-Cherent Limiter** — O(n)ピーク検出
- **True Peak Limiter** — インターサンプルピーク制限
- **LUFS正規化** — ITU-R BS.1770準拠
- **スペクトル分析** — Before/After比較
- **対応形式** — WAV, MP3, FLAC, OGG

## Docker デプロイ（推奨）

### クイックスタート

```bash
# ビルド & 起動（初回・コード更新後は --build 必須）
docker compose up -d --build

# コード更新後の強制再ビルド（キャッシュ回避）
docker compose up -d --build --force-recreate

# アクセス
open http://localhost:7860
```

### 初回起動時の注意

初回起動時にNumba JITコンパイルが走ります（約30秒）。2回目以降はキャッシュされるため高速に起動します。

### 永続ボリューム

```yaml
volumes:
  remaster-uploads:    # アップロードファイル
  remaster-outputs:    # リマスター済みファイル
  remaster-numba-cache: # Numba JIT キャッシュ
```

- `docker compose down` でデータは保持される
- `docker compose down -v` でデータも削除される
- 別PCに移行する場合: `docker compose up` するだけで同じ環境が構築される

### リソース制限

```yaml
deploy:
  resources:
    limits:
      memory: 4G
    reservations:
      memory: 512M
```

## ローカル実行

```bash
# 依存関係インストール
pip install -r requirements.txt

# 起動
python app.py
```

## API

| エンドポイント | Method | 用途 |
|---|---|---|
| `/` | GET | Web UI |
| `/upload` | POST | ファイルアップロード (単数) |
| `/upload-batch` | POST | 複数ファイル一括アップロード |
| `/analyze` | POST | スペクトル分析 |
| `/ai-optimize` | POST | AI分析 + 推奨設定生成 |
| `/remaster` | POST | リマスター実行 (単数) |
| `/batch-remaster` | POST | 一括リマスター |
| `/download/<filename>` | GET | ファイルダウンロード |
| `/download-batch` | POST | ZIP一括ダウンロード |

## 速度比較（Numba JIT）

| 関数 | 元 | Numba | スピードアップ |
|---|---|---|---|
| Phase Limit | 2.88s | 0.23s | 12.5x |
| Multiband | 2.16s | 0.28s | 7.7x |
| Limiter | 0.79s | 0.28s | 2.9x |
| **合計** | **5.84s** | **0.79s** | **7.4x** |

30秒音声 → **0.66秒** でリマスター完了。

## アーキテクチャ

```
┌─────────────────────────────────────┐
│  Remaster Studio (Docker)           │
│                                     │
│  Flask + Gunicorn (port 7860)       │
│  ├── app.py        (Web UI + API)   │
│  ├── dsp_optimized.py (Numba JIT)   │
│  └── remaster.py   (CLI)            │
│                                     │
│  DSP Pipeline:                      │
│  [0] Highpass FIR (fftconvolve)     │
│  [1] Parametric EQ Chain (4-band)   │
│  [2] Loudness Mapping Compress      │
│  [3] Mid/Side Compressor            │
│  [4] Multiband Compressor (3-band)  │
│  [5] Parallel Compressor            │
│  [6] Phase-Cohrent Limiter          │
│  [7] True Peak Limiter              │
│  [8] LUFS Normalization             │
│  [9] Stereo Enhancement             │
└─────────────────────────────────────┘
```

## ライセンス

MIT License
