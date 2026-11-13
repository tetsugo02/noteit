use crate::Endpoint;
use std::{
    ffi::OsString,
    io::ErrorKind,
    path::{Path, PathBuf},
    process::{Command, Output},
};

pub fn uv_candidates(home: Option<&Path>) -> Vec<PathBuf> {
    let mut candidates = vec![PathBuf::from("uv")];
    // Finder does not inherit the PATH configured by an interactive shell.
    if let Some(home) = home {
        candidates.push(home.join(".local/bin/uv"));
        candidates.push(home.join(".cargo/bin/uv"));
    }
    candidates.extend([
        PathBuf::from("/opt/homebrew/bin/uv"),
        PathBuf::from("/usr/local/bin/uv"),
    ]);
    candidates
}

fn endpoint(output: Output) -> Result<Endpoint, String> {
    if !output.status.success() {
        return Err(format!(
            "共通サービスの起動に失敗しました．\n{}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("共通サービスの起動応答を読み取れません．{error}"))
}

pub fn ensure_frontend(
    project: &Path,
    python: Option<OsString>,
    candidates: Vec<PathBuf>,
) -> Result<Endpoint, String> {
    if !project.is_dir() {
        return Err(format!(
            "Pythonサービスのディレクトリが見つかりません．\n{}\nリポジトリを元の場所に戻すか，この場所でアプリを再ビルドしてください．",
            project.display()
        ));
    }
    if let Some(python) = python {
        return endpoint(
            Command::new(&python)
                .args(["-m", "transcription_worker.frontend", "ensure"])
                .current_dir(project)
                .output()
                .map_err(|error| {
                    format!("NOTEIT_PYTHONの実行に失敗しました．\n{python:?}\n{error}")
                })?,
        );
    }
    for uv in candidates {
        match Command::new(&uv)
            .args(["run", "--frozen", "--directory"])
            .arg(project)
            .args(["noteit-front", "ensure"])
            .output()
        {
            Ok(output) => return endpoint(output),
            Err(error) if error.kind() == ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(format!(
                    "uvの実行に失敗しました．\n{}\n{error}",
                    uv.display()
                ));
            }
        }
    }
    Err("uvが見つかりません．uvをインストールするか，NOTEIT_PYTHONにサービスをインストールしたPythonの絶対パスを設定してください．".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_uv_reports_an_actionable_error() {
        let project = Path::new(env!("CARGO_MANIFEST_DIR"));
        let error = ensure_frontend(project, None, vec![project.join("missing-uv")])
            .err()
            .unwrap();
        assert!(error.contains("uvが見つかりません"));
        assert!(error.contains("NOTEIT_PYTHON"));
    }

    #[test]
    fn explicit_python_failure_does_not_fall_back_to_uv() {
        let project = Path::new(env!("CARGO_MANIFEST_DIR"));
        let error = ensure_frontend(project, Some(project.join("missing-python").into()), vec![])
            .err()
            .unwrap();
        assert!(error.contains("NOTEIT_PYTHONの実行に失敗"));
    }

    #[cfg(unix)]
    #[test]
    fn missing_path_entry_falls_back_to_an_absolute_executable() {
        use std::os::unix::fs::PermissionsExt;
        let directory = std::env::temp_dir().join(format!(
            "noteit launcher-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        let executable = directory.join("uv");
        std::fs::write(
            &executable,
            "#!/bin/sh\n[ \"$1\" = run ] && [ \"$2\" = --frozen ] && [ \"$3\" = --directory ] && [ -d \"$4\" ] && [ \"$5\" = noteit-front ] && [ \"$6\" = ensure ] || exit 2\nprintf '%s\\n' '{\"url\":\"http://127.0.0.1:8765\",\"token\":\"test\"}'\n",
        )
        .unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
        // The missing first candidate reproduces uv being absent from Finder's PATH.
        let result = ensure_frontend(
            &directory,
            None,
            vec![directory.join("missing-uv"), executable],
        );
        std::fs::remove_dir_all(directory).unwrap();
        let endpoint = result.unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(endpoint.url, "http://127.0.0.1:8765");
        assert_eq!(endpoint.token, "test");
    }
}
