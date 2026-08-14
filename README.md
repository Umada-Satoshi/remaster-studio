# 🎛️ Remaster Studio

高音質 重低音重視 リマスターツール with AI自動最適化

## 機能

- **18種のプロマスタリングプリセット** (Hip-Hop, EDM, Pop, Rock, Jazz等)
- **AI自動最適化** — ジャンル判定 + 最適設定の自動提案
- **バッチ処理** — 複数ファイルの一括リマスター
- **パラメトリックEQ** — Sub-Bass, Bass, Mid, High + カスタムバンド
- **マルチバンドコンプレッサー** — 3バンド (Low/Mid/High)
- **Brick-wall リミッター** — クリッピング防止
- **LUFS正規化** — ITU-R BS.1770準拠
- **スペクトル分析** — Before/After比較
- **対応形式** — WAV, MP3, FLAC, OGG

## クイックスタート (Docker)

```bash
# ビルド & 起動
docker compose up -d --build

# アクセス
open http://localhost:7860
```

## ローカル実行

```bash
# 依存関係インストール
pip install -r requirements.txt

# 起動
python app.py
```

## URL

- **メイン**: http://localhost:7860
- **ヘルスチェック**: http://localhost:7860/

## API

| エンドポイント | Method | 用途 |
|---|---|---|
| `/` | GET | Web UI |
| `/upload` | POST | ファイルアップロード (単数) |
| `/upload-batch` | POST | 複数ファイル一括アップロード |
| `/ai-optimize` | POST | AI分析 + 推奨設定生成 |
| `/remaster` | POST | リマスター実行 (単数) |
| `/batch-remaster` | POST | 一括リマスター |
| `/download/<filename>` | GET | ファイルダウンロード |

## Docker Compose設定

```yaml
services:
  remaster:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - remaster-uploads:/app/uploads
      - remaster-outputs:/app/outputs
    environment:
      - REMASTER_DATA_DIR=/app/data
    restart: unless-stopped
```

## 環境変数

| 変数 | 説明 | デフォルト |
|---|---|---|
| `REMASTER_DATA_DIR` | データ保存先ディレクトリ | 一時ディレクトリ |

## ライセンス

MIT License
