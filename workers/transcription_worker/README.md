# Transcription worker

開発起動，GUI・MCP，CLI，HTTP API，Dockerの手順は[リポジトリのREADME](../../README.md)を参照してください．

`TranscriptionService`はモデルを遅延読み込みし，単一のキューで推論します．CPU／CUDAはfaster-whisper，対応するMacの自動選択／Apple GPU指定はmlx-whisperです．話者分離は任意です．サービス・CLI・推論HTTP APIの`diarize`省略時は従来どおりtrue，GUI・新MCPではfalseを明示します．

```python
from transcription_worker.service import ServiceConfig, TranscriptionService

service = TranscriptionService(ServiceConfig.from_env())
job = service.submit("meeting.wav", diarize=False, language="ja")
service.wait(job.id, timeout=None)
result = service.get_result(job.id)
```

`TranscriptionWorker`を直接使う場合は，`diarizer=None`でWhisper単独認識できます．認識は音声全体を一度だけ実行し，単語を元の音声上の時刻とともに返します．`group_utterances`と`format_transcript`は話者ラベルnullと従来のラベル付き結果に対応しています．

デスクトップの`frontend`は設定・履歴・接続先をSQLiteで管理し，認証情報をmacOS Keychainから取得します．GUIはRust経由，MCPは`FrontendBackend`経由で同一サービスを呼び出します．ローカル推論はジョブごとの別プロセス`local_process`で行い，フロントにはモデルを読み込みません．モデルファイルはキャッシュしますが，ローカルのプロセス終了後に重みのメモリは保持しません．

HTTP推論サーバーは`ProcessTranscriptionService`を使い，ジョブごとに`inference_process`を起動します．子プロセスが終了してから結果を公開し，成功・失敗・サーバー停止時にモデルとVRAMを解放します．モデルのディスクキャッシュは再利用しますが，毎回メモリへの読み込みが発生します．待受プロセスには推論ライブラリをimportせず，モデルも保持しません．子プロセス用のSQLite・結果・デコード音声はジョブ専用の一時ディレクトリへ保存し，終了後に削除します．永続的なジョブ情報・結果は親プロセスが保存します．

直接CLIと，Pythonから直接使う`TranscriptionService`はプロセス内のモデルを再利用します．

デスクトップサービスとHTTP推論サーバーの`POST /jobs/{id}/cancel`は，待機中・実行中ジョブの停止を受け付けます．`cancelling`は停止要求中，`cancelled`は停止完了です．デスクトップサービスはリモートIDをアップロード前に保存し，停止要求もSQLiteに保持するため，通信断・サービス再起動後も同じサーバーの同じジョブへ停止を再送します．サーバー停止による`interrupted`と，利用者が要求した`cancelled`は区別します．完了ジョブに対する停止は結果を変更しません．
