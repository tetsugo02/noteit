#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use serde::Deserialize;
use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use tauri::{
    menu::{Menu, MenuItem, PredefinedMenuItem, Submenu},
    tray::TrayIconBuilder,
    Manager, State,
};

mod startup;

#[derive(Clone, Deserialize)]
struct Endpoint {
    url: String,
    token: String,
}
struct Desktop {
    endpoint: Endpoint,
    quitting: AtomicBool,
}

async fn call(
    endpoint: &Endpoint,
    method: &str,
    path: &str,
    body: Option<Value>,
) -> Result<Value, String> {
    if !path.starts_with('/') || path.contains("..") || path.starts_with("//") {
        return Err("Invalid service path".into());
    }
    let method = reqwest::Method::from_bytes(method.as_bytes()).map_err(|e| e.to_string())?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(35))
        .build()
        .map_err(|e| e.to_string())?;
    let mut request = client
        .request(method, format!("{}{}", endpoint.url, path))
        .bearer_auth(&endpoint.token);
    if let Some(body) = body {
        request = request.json(&body);
    }
    let response = request
        .send()
        .await
        .map_err(|e| format!("共通サービスへの接続に失敗しました．{e}"))?;
    let status = response.status();
    let data: Value = response.json().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(data["detail"]
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| data.to_string()));
    }
    Ok(data)
}

#[tauri::command]
async fn service_request(
    state: State<'_, Desktop>,
    method: String,
    path: String,
    body: Option<Value>,
) -> Result<Value, String> {
    call(&state.endpoint, &method, &path, body).await
}

#[tauri::command]
async fn choose_audio() -> Result<Option<String>, String> {
    tauri::async_runtime::spawn_blocking(|| {
        rfd::FileDialog::new()
            .set_title("音声ファイルを選択")
            .pick_file()
            .map(|p| p.to_string_lossy().into_owned())
    })
    .await
    .map_err(|e| e.to_string())
}

#[tauri::command]
async fn save_transcript(
    state: State<'_, Desktop>,
    job_id: String,
    format: String,
) -> Result<bool, String> {
    if !job_id.chars().all(|c| c.is_ascii_alphanumeric())
        || !["txt", "json"].contains(&format.as_str())
    {
        return Err("Invalid export".into());
    }
    let path = format!(
        "/jobs/{job_id}/{}",
        if format == "txt" { "text" } else { "result" }
    );
    let result = call(&state.endpoint, "GET", &path, None).await?;
    let content = if format == "txt" {
        format!("{}\n", result["text"].as_str().unwrap_or(""))
    } else {
        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
    };
    tauri::async_runtime::spawn_blocking(move || {
        if let Some(path) = rfd::FileDialog::new()
            .set_title("文字起こしを保存")
            .set_file_name(format!("{job_id}.{format}"))
            .add_filter(&format, &[&format])
            .save_file()
        {
            std::fs::write(path, content).map_err(|e| e.to_string())?;
            return Ok(true);
        }
        Ok(false)
    })
    .await
    .map_err(|e| e.to_string())?
}

fn quit(app: tauri::AppHandle) {
    if app.state::<Desktop>().quitting.swap(true, Ordering::SeqCst) {
        return;
    }
    tauri::async_runtime::spawn(async move {
        let state = app.state::<Desktop>();
        match call(&state.endpoint, "POST", "/shutdown", None).await {
            Ok(_) => {
                state.quitting.store(true, Ordering::SeqCst);
                app.exit(0);
            }
            Err(error) => {
                state.quitting.store(false, Ordering::SeqCst);
                rfd::AsyncMessageDialog::new()
                    .set_title("終了できませんでした")
                    .set_description(error)
                    .show()
                    .await;
            }
        }
    });
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            service_request,
            choose_audio,
            save_transcript
        ])
        .setup(|app| {
            let project = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../workers/transcription_worker");
            let home = app.path().home_dir().ok();
            let endpoint = match startup::ensure_frontend(
                &project,
                std::env::var_os("NOTEIT_PYTHON"),
                startup::uv_candidates(home.as_deref()),
            ) {
                Ok(endpoint) => endpoint,
                Err(error) => {
                    eprintln!("{error}");
                    rfd::MessageDialog::new()
                        .set_title("noteitを起動できませんでした")
                        .set_description(&error)
                        .set_level(rfd::MessageLevel::Error)
                        .show();
                    app.handle().exit(1);
                    return Ok(());
                }
            };
            app.manage(Desktop {
                endpoint,
                quitting: AtomicBool::new(false),
            });
            let show = MenuItem::with_id(app, "show", "noteitを表示", true, None::<&str>)?;
            let exit = MenuItem::with_id(
                app,
                "quit",
                "終了（ローカル推論を停止）",
                true,
                Some("CmdOrCtrl+Q"),
            )?;
            let app_menu = Menu::with_items(
                app,
                &[
                    &Submenu::with_items(app, "noteit", true, &[&show, &exit])?,
                    &Submenu::with_items(
                        app,
                        "ファイル",
                        true,
                        &[&PredefinedMenuItem::close_window(
                            app,
                            Some("ウィンドウを閉じる"),
                        )?],
                    )?,
                    &Submenu::with_items(
                        app,
                        "編集",
                        true,
                        &[
                            &PredefinedMenuItem::undo(app, None)?,
                            &PredefinedMenuItem::redo(app, None)?,
                            &PredefinedMenuItem::cut(app, None)?,
                            &PredefinedMenuItem::copy(app, None)?,
                            &PredefinedMenuItem::paste(app, None)?,
                            &PredefinedMenuItem::select_all(app, None)?,
                        ],
                    )?,
                ],
            )?;
            app.set_menu(app_menu)?;
            let menu = Menu::with_items(app, &[&show, &exit])?;
            let mut rgba = vec![0u8; 32 * 32 * 4];
            for y in 4..28 {
                for x in 5..27 {
                    let i = (y * 32 + x) * 4;
                    rgba[i..i + 4].copy_from_slice(&[65, 105, 225, 255]);
                }
            }
            TrayIconBuilder::new()
                .icon(tauri::image::Image::new_owned(rgba, 32, 32))
                .tooltip("noteit")
                .menu(&menu)
                .build(app)?;
            Ok(())
        })
        .on_menu_event(|app, event| match event.id.as_ref() {
            "show" => {
                if let Some(w) = app.get_webview_window("main") {
                    let _ = w.show();
                    let _ = w.set_focus();
                }
            }
            "quit" => quit(app.clone()),
            _ => {}
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .build(tauri::generate_context!())
        .expect("noteit desktop startup failed")
        .run(|app, event| {
            #[cfg(target_os = "macos")]
            if let tauri::RunEvent::Reopen { .. } = event {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }

            if let tauri::RunEvent::ExitRequested { api, .. } = event {
                if app
                    .try_state::<Desktop>()
                    .is_some_and(|state| !state.quitting.load(Ordering::SeqCst))
                {
                    api.prevent_exit();
                }
            }
        });
}
