# noteit

MacのデスクトップGUIとstdio MCPから利用する音声文字起こしツール．設定・履歴・結果を共通のPythonサービスで管理し，このMacのCPU／Apple GPU，または指定したHTTP推論サーバーで処理します．開発起動を対象とし，配布・署名・自動更新は含みません．

```text
Tauri 2 / React ─Rust IPC─┐
                        ├─ localhost FastAPI ─ SQLite / macOS Keychain
エージェント ─stdio MCP─┘         ├─ 別プロセス：MLX / faster-whisper CPU
                                 └─ HTTP：推論サーバー（CPU / CUDA）
```

## Macでのセットアップ

Python 3.12，uv，Node.js 22以上，Rust，Xcode Command Line Tools，FFmpegを用意してください．

```sh
brew install ffmpeg
# リモート接続のみの場合：モデルやtorchをインストールしません
uv sync --directory workers/transcription_worker --frozen

# ローカルCPUを追加
uv sync --directory workers/transcription_worker --frozen --extra cpu

# Apple Silicon：Apple GPUとCPUを追加
uv sync --directory workers/transcription_worker --frozen --extra cpu --extra mac-gpu

# 話者分離も使う場合
uv sync --directory workers/transcription_worker --frozen --extra cpu --extra mac-gpu --extra diarization

cd apps/desktop
npm ci
npm run tauri -- dev
```

GUIは初期状態で「このMac」「small」「話者分離OFF」です．音声ファイル，実行先，モデル，言語，必要なら話者分離・話者数を選び，「文字起こしを開始」を押してください．履歴から結果を表示し，TXT／JSONをネイティブの保存ダイアログで保存できます．モデル取得と読み込みを別の状態で表示します．取得率が得られない処理は状態と経過時間を表示します．

「自動」はMLXをインストール済みのApple SiliconではApple GPU，それ以外では利用可能なCUDA／CPUを選びます．「Apple GPU」「CPU」の明示指定が失敗した場合はエラーを表示します．実行途中で別エンジンへ切り替えません．Macの話者分離はCPU，Linux GPUサーバーではCUDAです．Whisper単独にはHugging Faceトークンは不要です．話者分離ではHugging Faceでpyannoteモデルの利用条件に同意し，GUIの設定にトークンを入力してください．

開発起動のTauriは`uv run --frozen --directory workers/transcription_worker noteit-front ensure`でPythonサービスを起動・再利用します．別のPython環境を使う場合は，パッケージをインストールしたPythonの絶対パスを`NOTEIT_PYTHON`に指定してください．推論ライブラリを追加・更新した際はトレイから明示終了し，再起動してください．

## 設定と終了

GUIの「設定と接続先」で，複数サーバーの名前・URL・APIトークンを登録できます．「接続テスト」は**保存済み**のプロファイルに対し認証とモデル・デバイス・話者分離対応を確認します．「既定の設定を保存」で実行先などの選択をMCPにも共有します．GUIは保存された設定と履歴を定期的に読み直します．未保存の編集中は設定の再読み込みを保留します．

ウィンドウを閉じると非表示になり，サービスと処理は継続します．メニューバーのnoteitアイコンから再表示・明示終了できます．アプリメニューの終了／Command-Qも明示終了です．明示終了はローカル推論のプロセスグループを停止します．サーバーに投入済みのリモートジョブは継続し，次回の共通サービス起動後に状態・結果を回収します．実行中ジョブのキャンセルUIはありません．

設定とジョブ履歴は`~/Library/Application Support/noteit/frontend.sqlite3`，認証情報はmacOS Keychainのサービス名`noteit`に保存します．`NOTEIT_DATA_DIR`でデータの保存先を変更できます．GUIとMCPで同じ値を使ってください．プロファイル変更・削除後も既存ジョブ用のURLとKeychain参照を保持するため，過去の認証情報は自動削除しません．

共通サービスは127.0.0.1の動的ポートで待ち受け，モード600の`endpoint.json`にクライアント用のランダム認証情報を保存します．起動ロックと実行中ロックで複数クライアントの重複起動を防ぎます．モデルはジョブ投入まで読み込みません．`service.log`にサービスの起動失敗や推論の診断を記録します．

## MCP

リポジトリの`.mcp.json`を利用するか，クライアントに次を登録してください．

```json
{"command":"uv","args":["run","--frozen","--directory","/absolute/path/noteit/workers/transcription_worker","noteit-mcp"]}
```

GUIが起動していなくても同じ共通サービスを起動します．MCPクライアントを切断しても共通サービスは継続します．

| ツール | 用途 |
|---|---|
| `list_models` | 現在の実行先のモデル，エンジン，デバイス，話者分離対応 |
| `get_transcription_settings` | GUIと共有する設定とプロファイル |
| `update_transcription_settings` | 既定の実行先・モデル・デバイス・言語などの更新 |
| `start_transcription` | 絶対パスの音声を投入．`diarize=false`が既定 |
| `wait_for_transcription` | 最大50秒待機し，状態・経過時間を取得 |
| `list_transcriptions` | 共通履歴の取得 |
| `get_transcript` | TXT形式で取得．話者分離OFFなら話者名を付けません |

`start_transcription`は`target`（`local`またはプロファイルID），`device`（`auto`／`mlx`／`cpu`），`whisper_model`，`language`，`diarize`，`num_speakers`などを指定できます．省略した実行先・モデル・デバイス・言語は共有設定から取得します．`language="auto"`で言語を自動検出します．`diarize`は共有設定にかかわらず省略時falseです．

従来の`NOTEIT_SERVER_URL`と`NOTEIT_API_TOKEN`は，共通サービスの**初回起動時**にプロファイルとして取り込みます．以降はGUI／共有設定を変更してください．CLIは引き続き毎回この環境変数を参照します．以前のローカル保存JSON（`NOTEIT_OUTPUT_DIR`，既定`~/.cache/noteit/transcripts`）は共通履歴に取り込みます．以前のリモートサーバーだけにあるジョブは，そのサーバーのAPIで取得できます．

## CLIと推論API

```sh
uv run --directory workers/transcription_worker transcribe-audio meeting.m4a -m small --device cpu --no-diarize -o transcript.txt
# CLIの省略時は従来どおり話者分離ON
uv run --directory workers/transcription_worker transcribe-audio meeting.m4a -n 2
# HTTP推論サーバーへ直接投入
NOTEIT_API_TOKEN=... uv run --directory workers/transcription_worker transcribe-audio meeting.m4a --server http://server:8765 --no-diarize
```

CLIは共有デスクトップサービスを経由せず，直接ローカル処理／HTTP処理を行います．HTTP APIの`POST /jobs`はmultipartの音声と`diarize`を受け付け，省略時trueです．`GET /models`には`engines`，`devices`，`diarization`を含みます．`GET /jobs/{id}`で状態，`GET /jobs/{id}/result`でJSONを取得します．`/health`以外はBearerトークンが必須です．

音声は全体を一度だけ認識し，共通の単語・秒単位の時刻形式へ変換します．faster-whisperはVAD後に元音声の時刻へ復元する実装を使用し，MLXは無音を連結除去せず元の音声を認識します．話者分離を使う場合も，話者ごとの断片を繰り返し認識しません．話者分離OFFでは話者ラベルと`diarizer_model`がnullになり，旧形式の話者ラベル付きJSONも読み込めます．

## 推論サーバーのDocker運用

LinuxのDocker EngineとComposeを用意してください．研究室LAN／VPN内で使用し，公開するホスト側IPを指定します．MacのApple GPUはDocker外で使用します．

```sh
cd infra/inference
cp .env.example .env
chmod 600 .env
# .envに十分長いNOTEIT_API_TOKENを設定
# NOTEIT_BIND_IPにLAN/VPNのインターフェースIPを設定（既定は127.0.0.1）
# 話者分離を使う場合だけHUGGINGFACE_ACCESS_TOKENも設定

docker compose up -d --build
docker compose ps
curl http://127.0.0.1:8765/health
```

GUIで`http://ホストIP:8765`と同じAPIトークンを登録し，接続テストを実行します．SSHトンネルを使う場合は`NOTEIT_BIND_IP=127.0.0.1`とし，Macから`ssh -N -L 8765:127.0.0.1:8765 server`で接続してください．

NVIDIA構成はホストのNVIDIAドライバーとNVIDIA Container Toolkitが必要です．

```sh
docker compose -f compose.yaml -f compose.cuda.yaml up -d --build
```

CPUイメージはPython 3.12.12，CUDAイメージは`nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`です．PyTorch／torchaudio 2.10.0，TorchCodec 0.10.0，faster-whisper 1.2.1／CTranslate2 4.8.0を固定し，CPU用とCUDA 12.8用のPyTorchインデックスを分離しています．CUDA構成のcuDNNは9系です．全依存の解決結果を`uv.lock`に保存しています．[PyTorchインデックスの設定](https://docs.astral.sh/uv/guides/integration/pytorch/)，[TorchCodecの互換表](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions)，[ComposeのGPU指定](https://docs.docker.com/compose/how-tos/gpu-support/)を参照してください．

サーバーは単一ユーザー・単一ワーカー・同時推論1件です．モデルキャッシュ，結果JSON，SQLiteのジョブ情報を`noteit-data`ボリュームの`/data`に保持します．再起動しても完了結果を取得できます．実行途中だったジョブは`interrupted`になり，手動再投入が必要です．アップロード音声は成功・失敗のいずれでも処理終了後に削除し，起動時に残ったアップロードとデコード用一時ファイルを削除します．ユーザーが選んだ元ファイルは削除しません．ヘルスチェックと`unless-stopped`を設定しています．

更新時はリポジトリを更新後，使用中と同じCompose引数で`up -d --build`を実行してください．停止は`docker compose down`です．データを残すため`down -v`は使わないでください．

## 検証

```sh
uv sync --directory workers/transcription_worker --frozen --extra cpu --extra mac-gpu --extra diarization
uv run --directory workers/transcription_worker pytest -q
uv run --directory workers/transcription_worker ruff check src tests
npm --prefix apps/desktop run build
cargo check --manifest-path apps/desktop/src-tauri/Cargo.toml
npm --prefix apps/desktop run tauri -- build --debug --no-bundle
```

CPU／Apple GPU／HTTPの実機確認と環境上の未検証事項は[検証記録](docs/verification.md)に記載します．LLM自動補正・要約API，追加学習，公開インターネット向けHTTPS，キャンセルUIは初版に含みません．MCPの`summarize_meeting`は呼び出し元LLM用のプロンプトであり，要約APIではありません．
