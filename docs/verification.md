# 検証記録

2026-11-03，Apple SiliconのmacOSで確認しました．実音声テストにはmacOSの`say`で生成した同じ短い英語音声を使用し，先頭に3秒の無音を追加しました．認識モデルは`tiny`，話者分離モデルは`community`，話者数は1です．

| 確認項目 | 結果 |
|---|---|
| Pythonの回帰・共通サービス契約テスト | 39件成功 |
| Ruff | 成功 |
| React / TypeScript / Viteの本番ビルド | 成功 |
| Rust `cargo check` | 成功 |
| Tauriのデバッグビルド（配布なし） | 成功 |
| ローカルCPU，Whisper単独 | 成功．faster-whisper / CPUを結果に記録 |
| ローカルCPU，話者分離あり | 成功．元音声上の時刻と話者ラベルを取得 |
| Apple GPU，Whisper単独 | 成功．mlx-whisper / mlxを結果に記録 |
| Apple GPU認識＋CPU話者分離 | 別プロセス経由で成功 |
| HTTP経由のCPU推論，Whisper単独／話者分離あり | 成功．認証エラーも401を確認 |
| HTTP推論サーバー再起動後の結果取得 | 両モードで成功 |
| GUIの音声選択・モデル選択・言語選択・処理開始 | ネイティブ画面で成功 |
| GUIの結果表示・TXT／JSON保存 | ネイティブ保存ダイアログで成功，保存内容も検証 |
| ウィンドウ閉鎖後のジョブ完了 | GUI投入後に閉鎖し，共通サービスの履歴で完了を確認 |
| GUIの明示終了 | アプリメニュー／Command-Qで終了し，サービスのendpoint削除を確認 |
| ローカル推論中の明示終了 | 子プロセスを実際に終了する契約テストで確認 |
| GUIとMCPの設定・ジョブ・結果の共有 | 同じFastAPIサービスに接続するMCP契約テストで確認 |
| 接続先編集・削除後の既存ジョブ | 元のURL・Keychain参照を維持することを確認 |
| 通信断・再起動後のリモートジョブ回収 | 元の接続先・リモートIDを維持し，入力音声なしで結果取得 |
| 認証情報の保存 | 実際のmacOS Keychainで検証用の値を書き込み・取得・削除 |
| モデル未導入のフロントのみの環境 | 別のvenvで起動・モデル一覧・終了に成功．推論ライブラリをimportしないことも確認 |
| 複数クライアントからの同時起動 | 別プロセス2件が同じendpointを再利用することを確認 |
| 旧JSONの読み込み・共有履歴への取り込み | 成功 |
| CLIのWhisper単独実行とTXT保存 | 実音声で成功 |
| CPU／CUDA Compose設定 | `docker compose config --quiet`成功 |
| Linux依存関係の分離 | CPU版にCUDA依存を含めず，CUDA版はPyTorch cu128／cuDNN 9を選択することをlockfileのexportで確認 |

Whisper単独の同一音声では，CPUが先頭語を約2.67秒，MLXが約2.90秒に配置しました．無音を除去した時刻へずれず，元音声の時間軸を保持しています．両エンジンで期待した発話を認識でき，話者ラベルと話者分離モデルはnullでした．短い合成音声での確認であり，長時間の日本語会議での認識精度・話者分離精度の評価ではありません．

HTTPの実音声検証で，`/tmp`と`/private/tmp`のようなシンボリックリンク経由のアップロードを早期削除する問題を検出し，パスの正規化と回帰テストを追加しました．軽量フロントの検証で，パッケージ初期化がpydubを先にimportする問題も検出し，公開import APIを維持した遅延importへ修正しました．

CPU用Dockerのビルド・起動・コンテナを再起動する検証は，Dockerデーモンが停止しているため未実施です．NVIDIA GPU実機がないため，CUDAコンテナ内での推論も未検証です．HTTP往復と永続化の実音声確認は，このMac上で推論サーバーを起動して行いました．トレイメニュー自体のクリックによる再表示は自動操作では未検証です．同じイベントハンドラーを使うアプリメニューとウィンドウの表示，終了を確認し，トレイの構成はTauriビルドで検証しています．

## 再確認用の音声

```sh
mkdir -p /tmp/noteit-audio-check
say -v Samantha -r 150 -o /tmp/noteit-audio-check/speech.aiff \
  'Hello. This is a short test of meeting transcription. We will finish the report tomorrow.'
ffmpeg -nostdin -loglevel error -y -i /tmp/noteit-audio-check/speech.aiff \
  -af adelay=3000 -ar 16000 -ac 1 /tmp/noteit-audio-check/speech.wav
```

GUIでこの音声を選び，`tiny`・英語・CPU／Apple GPUで実行してください．Whisper単独では話者名が表示されず，話者分離ONではラベルが表示されること，先頭の発話時刻が3秒付近になることを確認します．リモート構成でも同じ音声と設定を使用してください．初回はモデルのダウンロードが発生します．

## Dockerの未実施項目

[README](../README.md)の手順でDockerデーモンを起動した環境にデプロイした後，次を確認してください．

1. `/health`と認証付き`/models`を取得し，不正トークンが401になることを確認します．
2. 同じ音声を`diarize=false`／`true`で投入し，完了とTXT／JSON保存を確認します．
3. 完了ジョブIDを控えてコンテナを再起動し，同じIDの結果を取得します．
4. 実行中にコンテナを再起動し，そのジョブが`interrupted`になること，アップロードとデコード用一時ファイルが残らないことを確認します．
5. CUDA構成ではNVIDIA実機で同じ手順を行い，結果の`device`が`cuda`であることを確認します．
