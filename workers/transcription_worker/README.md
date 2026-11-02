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

デスクトップの`frontend`は設定・履歴・接続先をSQLiteで管理し，認証情報をmacOS Keychainから取得します．GUIはRust経由，MCPは`FrontendBackend`経由で同一サービスを呼び出します．ローカル推論はジョブごとの別プロセス`local_process`で行い，フロントにはモデルを読み込みません．モデルファイルはキャッシュしますが，ローカルのプロセス終了後に重みのメモリは保持しません．推論サーバーと直接CLIではプロセス内のモデルを再利用します．
