import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { invoke } from '@tauri-apps/api/core';
import './style.css';

type Settings = {target: string; device: string; whisper_model: string; language: string | null; diarize: boolean};
type Profile = {id: string; name: string; url: string};
type Catalog = {whisper_models: {name: string}[]; engines: {name: string; devices: string[]}[]; devices: string[]; diarization: boolean};
type Job = {job_id: string; status: string; audio_path: string; target: {type: string; name: string; url?: string}; whisper_model: string; engine: string | null; device: string | null; diarizer_model: string | null; elapsed_seconds: number; segments_done: number; segments_total: number | null; error: string | null};
type Transcript = {utterances: {speaker_label: string | null; text: string; start_time_seconds: number}[]};
const finished = ['done', 'failed', 'interrupted'];
const states: Record<string, string> = {queued:'待機中', connecting:'接続中', uploading:'アップロード中', downloading_model:'モデル取得・キャッシュ確認中', loading_models:'モデル読み込み中', diarizing:'話者分離中', transcribing:'音声認識中', connection_lost:'再接続待ち', done:'完了', failed:'失敗', interrupted:'中断'};
const api = <T,>(method: string, path: string, body?: unknown): Promise<T> => invoke('service_request', {method, path, body: body ?? null});
const stamp = (n: number) => [Math.floor(n / 3600), Math.floor(n / 60) % 60, Math.floor(n) % 60].map(v => String(v).padStart(2,'0')).join(':');

function App() {
  const [settings, setSettings] = useState<Settings>({target:'local', device:'auto', whisper_model:'small', language:'ja', diarize:false});
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [audio, setAudio] = useState('');
  const [speakers, setSpeakers] = useState('');
  const [selected, setSelected] = useState('');
  const [result, setResult] = useState<Transcript | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);
  const [dirty, setDirty] = useState(false);
  const dirtyRef = useRef(false);
  const [hfToken, setHfToken] = useState('');
  const [profile, setProfile] = useState({id:'', name:'', url:'', token:''});
  const [connection, setConnection] = useState('');
  const selectedJob = jobs.find(job => job.job_id === selected);
  const change = (update: Partial<Settings>) => { dirtyRef.current = true; setDirty(true); setSettings(s => ({...s, ...update})); };

  useEffect(() => {
    let alive = true;
    const refresh = async () => {
      try {
        const [history, defaults, endpoints] = await Promise.all([api<Job[]>('GET','/jobs'), api<Settings>('GET','/settings'), api<Profile[]>('GET','/profiles')]);
        if (!alive) return;
        setJobs(history); setProfiles(endpoints);
        if (!dirtyRef.current) setSettings(defaults);
      } catch (e) { if (alive) setError(String(e)); }
    };
    void refresh(); const timer = setInterval(() => void refresh(), 1500);
    return () => { alive = false; clearInterval(timer); };
  }, []);

  useEffect(() => {
    let alive = true; setCatalog(null);
    api<Catalog>('GET', `/models?target=${encodeURIComponent(settings.target)}`).then(c => {if (alive) setCatalog(c);}).catch(e => {if (alive) setError(String(e));});
    return () => {alive = false;};
  }, [settings.target, profiles.find(p => p.id === settings.target)?.url]);

  useEffect(() => {
    let alive = true; setResult(null);
    if (selectedJob?.status === 'done') api<Transcript>('GET', `/jobs/${selected}/result`).then(r => {if (alive) setResult(r);}).catch(e => {if (alive) setError(String(e));});
    return () => {alive = false;};
  }, [selected, selectedJob?.status]);

  async function action(work: () => Promise<void>) {
    setError(''); setNotice(''); setBusy(true);
    try { await work(); } catch (e) {setError(String(e));} finally {setBusy(false);}
  }
  async function saveSettings() {
    await api('PUT','/settings', {...settings, ...(hfToken ? {hf_token:hfToken} : {})});
    setHfToken(''); dirtyRef.current = false; setDirty(false);
  }
  async function saveProfile() {
    const saved = await api<Profile>('POST','/profiles', {...profile, id:profile.id || null, token:profile.token || null});
    setProfiles(await api<Profile[]>('GET','/profiles'));
    setProfile({id:saved.id, name:saved.name, url:saved.url, token:''});
    setNotice('接続先を保存しました．APIトークンはKeychainに保存されています．');
  }

  return <div className="shell">
    <header><div className="brand">n<span>noteit</span></div><p>音声を，読み返せる記録へ．</p><span className="service">共通サービス接続</span></header>
    {error && <div role="alert" className="message error">{error}<button onClick={() => setError('')}>閉じる</button></div>}
    {notice && <div role="status" className="message notice">{notice}</div>}
    <main>
      <aside>
        <section><h2>文字起こし</h2><label>音声ファイル</label><button className="file" disabled={busy} onClick={() => void action(async () => {const path = await invoke<string | null>('choose_audio'); if (path) setAudio(path);})}>{audio.split('/').pop() || 'ファイルを選択'}</button>{audio && <p className="path">{audio}</p>}
          <label htmlFor="target">実行先</label><select id="target" value={settings.target} onChange={e => change({target:e.target.value})}><option value="local">このMac</option>{profiles.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}</select>
          {settings.target === 'local' && <><label htmlFor="device">デバイス</label><select id="device" value={settings.device} onChange={e => change({device:e.target.value})}><option value="auto">自動</option><option value="mlx">Apple GPU</option><option value="cpu">CPU</option></select></>}
          <label htmlFor="model">認識モデル</label><select id="model" value={settings.whisper_model} onChange={e => change({whisper_model:e.target.value})}>{(catalog?.whisper_models || ['tiny','base','small','medium','turbo','large'].map(name=>({name}))).map(m => <option key={m.name}>{m.name}</option>)}</select>
          {catalog && <p className="hint">利用可能：{catalog.engines.map(e => `${e.name}（${e.devices.join(' / ')}）`).join('，') || '推論用の依存関係を追加してください．'}</p>}
          <label htmlFor="language">言語</label><select id="language" value={settings.language ?? 'auto'} onChange={e=>change({language:e.target.value === 'auto' ? null : e.target.value})}><option value="ja">日本語</option><option value="en">英語</option><option value="auto">自動検出</option></select>
          <label className="check"><input type="checkbox" checked={settings.diarize} onChange={e => change({diarize:e.target.checked})}/>話者分離を有効にする</label>
          {settings.diarize && <><label htmlFor="speakers">話者数（任意）</label><input id="speakers" type="number" min="1" step="1" value={speakers} onChange={e => setSpeakers(e.target.value)}/>{catalog && !catalog.diarization && <p className="hint">話者分離の依存関係がありません．セットアップ手順を確認してください．</p>}</>}
          <button className="primary" disabled={!audio || busy} onClick={() => void action(async () => {
            if (settings.diarize && speakers && (!Number.isInteger(Number(speakers)) || Number(speakers)<1)) throw new Error('話者数は1以上の整数で入力してください．');
            await saveSettings();
            const job = await api<Job>('POST','/jobs',{...settings, audio_path:audio, num_speakers:settings.diarize && speakers ? Number(speakers) : null});
            setSelected(job.job_id); setJobs(await api<Job[]>('GET','/jobs'));
          })}>文字起こしを開始</button>
          <p className="hint">初回はモデルを取得します．ウィンドウを閉じても処理は継続します．</p>
        </section>
        <details><summary>設定と接続先</summary><section className="settings">
          <label htmlFor="hf">Hugging Faceトークン</label><input id="hf" type="password" autoComplete="off" value={hfToken} onChange={e=>setHfToken(e.target.value)} placeholder="話者分離を使う場合に設定"/>
          <button disabled={busy || (!dirty && !hfToken)} onClick={()=>void action(async()=>{await saveSettings(); setNotice('既定の設定を保存しました．');})}>既定の設定を保存</button>
          <p className="hint">トークンはKeychainに保存します．空欄では既存の値を維持します．</p>
          <hr/><h3>リモートサーバー</h3><label htmlFor="profile">接続先を編集</label><select id="profile" value={profile.id} onChange={e=>{const p=profiles.find(p=>p.id===e.target.value); setProfile(p ? {...p,token:''} : {id:'',name:'',url:'',token:''}); setConnection('');}}><option value="">新規登録</option>{profiles.map(p=><option key={p.id} value={p.id}>{p.name}</option>)}</select>
          <label htmlFor="profile-name">名前</label><input id="profile-name" value={profile.name} onChange={e=>setProfile({...profile,name:e.target.value})}/>
          <label htmlFor="profile-url">URL</label><input id="profile-url" type="url" placeholder="http://server:8765" value={profile.url} onChange={e=>setProfile({...profile,url:e.target.value})}/>
          <label htmlFor="api-token">APIトークン</label><input id="api-token" type="password" autoComplete="off" placeholder={profile.id ? '空欄で現在の値を維持' : ''} value={profile.token} onChange={e=>setProfile({...profile,token:e.target.value})}/>
          <div className="row"><button disabled={busy || !profile.name || !profile.url} onClick={()=>void action(saveProfile)}>保存</button><button disabled={busy || !profile.id} onClick={()=>void action(async()=>{const c=await api<Catalog>('POST',`/profiles/${profile.id}/test`); setConnection(`認証成功．モデル：${c.whisper_models.map(m=>m.name).join(' / ')}．デバイス：${c.devices.join(' / ')}．話者分離：${c.diarization ? '対応' : '非対応'}．`);})}>接続テスト</button></div>
          {connection && <p role="status" className="hint">{connection}</p>}
          {profile.id && <button className="danger" disabled={busy} onClick={()=>void action(async()=>{await api('DELETE',`/profiles/${profile.id}`); setProfiles(await api<Profile[]>('GET','/profiles')); setSettings(await api<Settings>('GET','/settings')); setProfile({id:'',name:'',url:'',token:''}); setNotice('接続先を削除しました．既存ジョブの接続先は保持されます．');})}>接続先を削除</button>}
        </section></details>
      </aside>
      <article>
        <section className="history"><div className="section-title"><h2>ジョブ履歴</h2><span>{jobs.length}件</span></div>{!jobs.length && <div className="empty">音声ファイルを選択して，最初の文字起こしを開始してください．</div>}
          {jobs.map(job=><button key={job.job_id} className={`job ${selected===job.job_id?'selected':''}`} onClick={()=>setSelected(job.job_id)}>
            <span><strong>{job.audio_path.split('/').pop()}</strong><small>{job.target.name} · {job.engine || 'エンジン未確定'} · {job.whisper_model}{job.device ? ` · ${job.device}` : ''}{job.diarizer_model ? ' · 話者分離あり' : ''}</small></span><span className="job-state"><b>{states[job.status] || job.status}</b><small>{stamp(job.elapsed_seconds)}</small></span>
          </button>)}
        </section>
        <section className="transcript"><div className="section-title"><h2>結果</h2><div className="row"><button disabled={!result || busy} onClick={()=>void action(async()=>{if(await invoke('save_transcript',{jobId:selected,format:'txt'}))setNotice('TXTを保存しました．');})}>TXT保存</button><button disabled={!result || busy} onClick={()=>void action(async()=>{if(await invoke('save_transcript',{jobId:selected,format:'json'}))setNotice('JSONを保存しました．');})}>JSON保存</button></div></div>
          {selectedJob?.error && <p role="alert" className="error detail-error">{selectedJob.error}</p>}
          {selectedJob && !finished.includes(selectedJob.status) && <div role="status" className="progress"><span className="spinner"/>{states[selectedJob.status] || selectedJob.status} · 経過 {stamp(selectedJob.elapsed_seconds)}{selectedJob.segments_total ? ` · ${selectedJob.segments_done}/${selectedJob.segments_total}` : ''}</div>}
          {result ? (result.utterances.length ? result.utterances.map((u,i)=><div className="utterance" key={i}><time>{stamp(u.start_time_seconds)}</time><div>{u.speaker_label && <b>{u.speaker_label}</b>}<p>{u.text}</p></div></div>) : <div className="empty">音声から発話は検出されませんでした．</div>) : <div className="empty">履歴を選択すると，完了した文字起こしを表示します．</div>}
        </section>
      </article>
    </main>
    <footer>MCPと設定・履歴を共有しています．明示的な終了はメニューバーのnoteitアイコンから行えます．</footer>
  </div>;
}
createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
