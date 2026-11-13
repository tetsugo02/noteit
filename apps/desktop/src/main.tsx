import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { invoke } from '@tauri-apps/api/core';
import './style.css';

type Settings = { target: string; device: string; whisper_model: string; language: string | null; diarize: boolean };
type Profile = { id: string; name: string; url: string };
type ProfileDraft = Profile & { token: string };
type Catalog = { whisper_models: { name: string }[]; devices: string[]; diarization: boolean };
type Job = { job_id: string; status: string; audio_path: string; target: { type: string; name: string }; elapsed_seconds: number; segments_done: number; segments_total: number | null; error: string | null };
type Transcript = { utterances: { speaker_label: string | null; text: string; start_time_seconds: number }[] };
const finished = ['done', 'failed', 'interrupted', 'cancelled'];
const states: Record<string, string> = { queued: '待機中', connecting: '接続中', uploading: 'アップロード中', downloading_model: 'モデルを準備中', loading_models: 'モデルを読み込み中', diarizing: '話者を分離中', transcribing: '文字起こし中', connection_lost: '再接続中', cancelling: '停止中', done: '完了', failed: '失敗', interrupted: '中断', cancelled: '停止済み' };
const api = <T,>(method: string, path: string, body?: unknown): Promise<T> => invoke('service_request', { method, path, body: body ?? null });
const stamp = (seconds: number) => [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, Math.floor(seconds) % 60].map(v => String(v).padStart(2, '0')).join(':');
const filename = (path: string) => path.split('/').pop() || path;
const blankProfile = (): ProfileDraft => ({ id: '', name: '', url: '', token: '' });
const paths = {
  plus: 'M12 5v14M5 12h14', close: 'm6 6 12 12M18 6 6 18',
  server: 'M4 4h16v6H4zM4 14h16v6H4zM7 7h.01M7 17h.01',
  settings: 'M9.5 4.5 10 2h4l.5 2.5 2 .9 2.3-.8 2 3.4-1.8 1.7v2.6l1.8 1.7-2 3.4-2.3-.8-2 .9L14 20h-4l-.5-2.5-2-.9-2.3.8-2-3.4L5 12.3V9.7L3.2 8l2-3.4 2.3.8zM12 8a3 3 0 1 0 0 6 3 3 0 0 0 0-6',
  audio: 'M9 18V5l12-2v13M9 9l12-2M9 18c0 1.7-1.8 3-4 3s-4-1.3-4-3 1.8-3 4-3 4 1.3 4 3ZM21 16c0 1.7-1.8 3-4 3s-4-1.3-4-3 1.8-3 4-3 4 1.3 4 3Z',
  download: 'M12 3v12m-5-5 5 5 5-5M5 16v5h14v-5', arrow: 'm8 10 4 4 4-4', stop: 'M6 6h12v12H6z', check: 'm5 12 4 4L19 6',
};
function Icon({ name, size = 20 }: { name: keyof typeof paths; size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>;
}
function Modal({ title, open, onClose, children, wide = false }: { title: string; open: boolean; onClose: () => void; children: React.ReactNode; wide?: boolean }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => { const dialog = ref.current; if (open && dialog && !dialog.open) dialog.showModal(); if (!open && dialog?.open) dialog.close(); }, [open]);
  return <dialog ref={ref} className={`modal ${wide ? 'modal-wide' : ''}`} aria-label={title} onCancel={e => { e.preventDefault(); onClose(); }} onClick={e => { if (e.target === e.currentTarget) { const rect = e.currentTarget.getBoundingClientRect(); if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) onClose(); } }}><div className="modal-heading"><h2>{title}</h2><button className="icon-button" aria-label="閉じる" onClick={onClose}><Icon name="close" /></button></div>{children}</dialog>;
}

function App() {
  const [settings, setSettings] = useState<Settings>({ target: 'local', device: 'auto', whisper_model: 'small', language: 'ja', diarize: false });
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [audio, setAudio] = useState('');
  const [speakers, setSpeakers] = useState('');
  const [selected, setSelected] = useState('');
  const [result, setResult] = useState<Transcript | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [stopping, setStopping] = useState('');
  const dirty = useRef(false);
  const [hostsOpen, setHostsOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [hfToken, setHfToken] = useState('');
  const [profile, setProfile] = useState<ProfileDraft>(blankProfile);
  const [modalError, setModalError] = useState('');
  const selectedJob = jobs.find(job => job.job_id === selected);
  const running = !!selectedJob && !finished.includes(selectedJob.status);
  const change = (update: Partial<Settings>) => { dirty.current = true; setSettings(s => ({ ...s, ...update })); };
  const refreshJobs = async () => setJobs(await api<Job[]>('GET', '/jobs'));

  useEffect(() => {
    let alive = true; let refreshing = false;
    const refresh = async () => {
      if (refreshing) return;
      refreshing = true;
      try {
        const [history, defaults, endpoints] = await Promise.all([api<Job[]>('GET', '/jobs'), api<Settings>('GET', '/settings'), api<Profile[]>('GET', '/profiles')]);
        if (!alive) return;
        setConnected(true); setJobs(history); setProfiles(endpoints);
        setSelected(current => current || history[0]?.job_id || '');
        if (!dirty.current) setSettings(defaults);
      } catch (e) { if (alive) { setConnected(false); setError(String(e)); } }
      finally { refreshing = false; }
    };
    void refresh(); const timer = setInterval(() => void refresh(), 1000);
    return () => { alive = false; clearInterval(timer); };
  }, []);
  useEffect(() => {
    let alive = true; setCatalog(null);
    api<Catalog>('GET', `/models?target=${encodeURIComponent(settings.target)}`).then(value => { if (alive) setCatalog(value); }).catch(e => { if (alive) setError(String(e)); });
    return () => { alive = false; };
  }, [settings.target, profiles.find(p => p.id === settings.target)?.url]);
  useEffect(() => {
    let alive = true; setResult(null);
    if (selectedJob?.status === 'done') api<Transcript>('GET', `/jobs/${selected}/result`).then(value => { if (alive) setResult(value); }).catch(e => { if (alive) setError(String(e)); });
    return () => { alive = false; };
  }, [selected, selectedJob?.status]);
  useEffect(() => { if (!notice) return; const timer = setTimeout(() => setNotice(''), 3500); return () => clearTimeout(timer); }, [notice]);

  async function action(work: () => Promise<void>, modal = false) {
    if (busy) return;
    if (modal) setModalError(''); else setError('');
    setBusy(true);
    try { await work(); } catch (e) { if (modal) setModalError(String(e)); else setError(String(e)); } finally { setBusy(false); }
  }
  async function saveSettings(value = settings) {
    await api('PUT', '/settings', { ...value, ...(hfToken ? { hf_token: hfToken } : {}) });
    setHfToken(''); dirty.current = false;
  }
  function openHosts() {
    const current = profiles.find(p => p.id === settings.target) || profiles[0];
    setProfile(current ? { ...current, token: '' } : blankProfile());
    setModalError(''); setHostsOpen(true);
  }
  async function connectProfile(event: React.FormEvent) {
    event.preventDefault();
    await action(async () => {
      const saved = await api<Profile>('POST', '/profiles', { ...profile, name: profile.name.trim(), url: profile.url.trim(), id: profile.id || null, token: profile.token || null });
      setProfile({ ...saved, token: '' }); setProfiles(await api<Profile[]>('GET', '/profiles'));
      await api<Catalog>('POST', `/profiles/${saved.id}/test`);
      const next = { ...settings, target: saved.id };
      await saveSettings(next); setSettings(next); setHostsOpen(false); setNotice('接続しました');
    }, true);
  }
  async function start() {
    await action(async () => {
      if (settings.diarize && speakers && (!Number.isInteger(Number(speakers)) || Number(speakers) < 1)) throw new Error('話者数は1以上の整数で入力してください．');
      await saveSettings();
      const job = await api<Job>('POST', '/jobs', { ...settings, audio_path: audio, num_speakers: settings.diarize && speakers ? Number(speakers) : null });
      setSelected(job.job_id); await refreshJobs();
    });
  }
  async function stop(job: Job) {
    setStopping(job.job_id); setError('');
    try { const update = await api<Job>('POST', `/jobs/${job.job_id}/cancel`); setJobs(current => current.map(item => item.job_id === job.job_id ? update : item)); await refreshJobs(); }
    catch (e) { setError(String(e)); } finally { setStopping(''); }
  }

  return <div className="shell">
    <header className="app-header"><div className="brand"><span className="brand-mark" aria-hidden="true">n</span><span>noteit</span><span className={`connection-dot ${connected ? 'online' : ''}`} role="status" aria-label={connected ? 'サービス接続済み' : 'サービス未接続'} /></div><div className="header-actions"><button className="button-quiet" onClick={openHosts}><Icon name="server" />接続先</button><button className="icon-button" aria-label="設定" onClick={() => { setModalError(''); setSettingsOpen(true); }}><Icon name="settings" /></button></div></header>
    {error && <div role="alert" className="banner error"><span>{error}</span><button className="icon-button" aria-label="エラーを閉じる" onClick={() => setError('')}><Icon name="close" size={18} /></button></div>}
    {notice && <div role="status" className="toast"><Icon name="check" size={18} />{notice}</div>}
    <main className="workspace"><aside className="sidebar">
      <section className="composer card" aria-label="文字起こし設定"><h1>文字起こし</h1>
        <button className={`file-picker ${audio ? 'has-file' : ''}`} disabled={busy} title={audio || undefined} onClick={() => void action(async () => { const path = await invoke<string | null>('choose_audio'); if (path) setAudio(path); })}><span className="file-icon"><Icon name="audio" size={23} /></span><span className="file-label">{audio ? filename(audio) : '音声ファイルを選択'}</span><Icon name="plus" size={18} /></button>
        <div className="field"><div className="label-row"><label htmlFor="target">実行先</label><button className="inline-button" aria-label="接続先を追加" onClick={() => { setProfile(blankProfile()); setModalError(''); setHostsOpen(true); }}><Icon name="plus" size={16} /></button></div><div className="select-wrap"><select id="target" value={settings.target} onChange={e => change({ target: e.target.value })}><option value="local">このMac</option>{profiles.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}</select><Icon name="arrow" size={18} /></div></div>
        <div className="field-pair"><div className="field"><label htmlFor="model">モデル</label><div className="select-wrap"><select id="model" value={settings.whisper_model} onChange={e => change({ whisper_model: e.target.value })}>{(catalog?.whisper_models || ['tiny', 'base', 'small', 'medium', 'turbo', 'large'].map(name => ({ name }))).map(model => <option key={model.name}>{model.name}</option>)}</select><Icon name="arrow" size={18} /></div></div><div className="field"><label htmlFor="language">言語</label><div className="select-wrap"><select id="language" value={settings.language ?? 'auto'} onChange={e => change({ language: e.target.value === 'auto' ? null : e.target.value })}><option value="ja">日本語</option><option value="en">英語</option><option value="auto">自動</option></select><Icon name="arrow" size={18} /></div></div></div>
        <button className="toggle-row" role="switch" aria-checked={settings.diarize} onClick={() => change({ diarize: !settings.diarize })}><span>話者分離</span><span className="switch-track"><span /></span></button>
        {settings.diarize && <div className="field speaker-field"><label htmlFor="speakers">話者数</label><input id="speakers" type="number" min="1" step="1" placeholder="自動" value={speakers} onChange={e => setSpeakers(e.target.value)} /></div>}
        <button className="button-primary start-button" disabled={!audio || busy || !connected} onClick={() => void start()}><Icon name="audio" size={19} />{busy ? '準備中…' : '文字起こしを開始'}</button>
      </section>
      <section className="history card" aria-label="履歴"><div className="history-heading"><h2>履歴</h2><span>{jobs.length}</span></div><div className="history-list">{jobs.length ? jobs.map(job => <button key={job.job_id} className={`job ${selected === job.job_id ? 'selected' : ''}`} aria-pressed={selected === job.job_id} onClick={() => setSelected(job.job_id)}><span className="job-title">{filename(job.audio_path)}</span><span className="job-meta"><span>{job.target.name}</span><span className={`job-status status-${job.status}`}>{states[job.status] || job.status}</span></span></button>) : <div className="history-empty">履歴はありません</div>}</div></section>
    </aside>
    <section className="transcript card" aria-label="文字起こし結果"><div className="result-heading"><div className="result-title"><span className="eyebrow">結果</span><h2 title={selectedJob?.audio_path}>{selectedJob ? filename(selectedJob.audio_path) : '文字起こし結果'}</h2></div><div className="export-actions"><button disabled={!result || busy} aria-label="TXTを保存" onClick={() => void action(async () => { if (await invoke('save_transcript', { jobId: selected, format: 'txt' })) setNotice('TXTを保存しました'); })}><Icon name="download" size={17} />TXT</button><button disabled={!result || busy} aria-label="JSONを保存" onClick={() => void action(async () => { if (await invoke('save_transcript', { jobId: selected, format: 'json' })) setNotice('JSONを保存しました'); })}><Icon name="download" size={17} />JSON</button></div></div>
      {selectedJob && <div className={`activity ${running ? 'activity-running' : ''}`}><div className="activity-info">{running && <span className="spinner" />}<span className={`job-status status-${selectedJob.status}`}>{states[selectedJob.status] || selectedJob.status}</span><time>{stamp(selectedJob.elapsed_seconds)}</time>{!!selectedJob.segments_total && running && <span className="segment-count">{selectedJob.segments_done}/{selectedJob.segments_total}</span>}</div>{running && <button className="stop-button" disabled={selectedJob.status === 'cancelling' || stopping === selectedJob.job_id} onClick={() => void stop(selectedJob)}><Icon name="stop" size={16} />{selectedJob.status === 'cancelling' || stopping === selectedJob.job_id ? '停止中…' : '停止'}</button>}</div>}
      {selectedJob?.error && <div role="alert" className="result-error">{selectedJob.error}</div>}
      <div className="transcript-body">{result ? (result.utterances.length ? result.utterances.map((utterance, index) => <div className="utterance" key={index}><time>{stamp(utterance.start_time_seconds)}</time><div>{utterance.speaker_label && <span className="speaker">{utterance.speaker_label}</span>}<p>{utterance.text}</p></div></div>) : <div className="empty-state"><Icon name="audio" size={34} /><p>発話が見つかりませんでした</p></div>) : <div className="empty-state"><span className={running ? 'waveform animated' : 'waveform'} aria-hidden="true">{[0, 1, 2, 3, 4].map(i => <i key={i} />)}</span><p>{selectedJob?.status === 'cancelled' ? '文字起こしを停止しました' : running ? states[selectedJob!.status] : selectedJob?.status === 'failed' ? '文字起こしに失敗しました' : selectedJob?.status === 'interrupted' ? '文字起こしが中断されました' : '音声ファイルを選択してください'}</p></div>}</div>
    </section></main>
    <Modal title="接続先" open={hostsOpen} onClose={() => { if (!busy) setHostsOpen(false); }} wide><div className="host-layout"><nav className="host-list" aria-label="登録済みの接続先">{profiles.map(item => <button key={item.id} className={`host-item ${profile.id === item.id ? 'selected' : ''}`} disabled={busy} onClick={() => { setProfile({ ...item, token: '' }); setModalError(''); }}><Icon name="server" size={19} /><span><strong>{item.name}</strong><small>{item.url}</small></span></button>)}<button className={`host-add ${!profile.id ? 'selected' : ''}`} disabled={busy} onClick={() => { setProfile(blankProfile()); setModalError(''); }}><Icon name="plus" size={19} />新規登録</button></nav>
      <form className="host-form" onSubmit={e => void connectProfile(e)}><div className="field"><label htmlFor="host-name">名前</label><input id="host-name" autoFocus required autoComplete="off" placeholder="GPUサーバー" value={profile.name} disabled={busy} onChange={e => setProfile({ ...profile, name: e.target.value })} /></div><div className="field"><label htmlFor="host-url">URL</label><input id="host-url" required type="url" placeholder="http://100.x.x.x:8765" value={profile.url} disabled={busy} onChange={e => setProfile({ ...profile, url: e.target.value })} /></div><div className="field"><label htmlFor="host-token">APIトークン</label><input id="host-token" type="password" required={!profile.id} autoComplete="off" placeholder={profile.id ? '変更する場合のみ入力' : ''} value={profile.token} disabled={busy} onChange={e => setProfile({ ...profile, token: e.target.value })} /></div>{modalError && <div role="alert" className="modal-error">{modalError}</div>}<div className="modal-actions">{profile.id && <button type="button" className="button-danger" disabled={busy} onClick={() => void action(async () => { await api('DELETE', `/profiles/${profile.id}`); setProfiles(await api<Profile[]>('GET', '/profiles')); if (settings.target === profile.id) { const next = { ...settings, target: 'local' }; await saveSettings(next); setSettings(next); } setProfile(blankProfile()); setNotice('接続先を削除しました'); }, true)}>削除</button>}<button type="submit" className="button-primary" disabled={busy || !profile.name.trim() || !profile.url.trim() || (!profile.id && !profile.token)}>{busy ? <><span className="spinner" />接続中…</> : '保存して接続'}</button></div></form>
    </div></Modal>
    <Modal title="設定" open={settingsOpen} onClose={() => { if (!busy) setSettingsOpen(false); }}><form className="preferences" onSubmit={e => { e.preventDefault(); void action(async () => { await saveSettings(); setSettingsOpen(false); setNotice('設定を保存しました'); }, true); }}><div className="field"><label htmlFor="device">ローカルのデバイス</label><div className="select-wrap"><select id="device" value={settings.device} onChange={e => change({ device: e.target.value })}><option value="auto">自動</option><option value="mlx">Apple GPU</option><option value="cpu">CPU</option></select><Icon name="arrow" size={18} /></div></div><div className="field"><label htmlFor="hf-token">Hugging Faceトークン</label><input id="hf-token" type="password" autoComplete="off" value={hfToken} onChange={e => setHfToken(e.target.value)} placeholder="話者分離で使用" /></div>{modalError && <div role="alert" className="modal-error">{modalError}</div>}<div className="modal-actions"><button type="submit" className="button-primary" disabled={busy}>保存</button></div></form></Modal>
  </div>;
}
createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>);
