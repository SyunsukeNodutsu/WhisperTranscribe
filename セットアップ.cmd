@echo off
rem One-time setup: create the venv and install libraries. Run before the first use.
rem
rem NOTE: keep this file as Shift-JIS (CP932) with CRLF line endings.
rem   - LF-only line endings break cmd's parser once the script contains if-blocks
rem     or goto: lines get chopped mid-word and you get bogus "is not recognized"
rem     errors. The other .cmd files here are LF but short enough to survive.
rem   - Japanese text must be CP932 because a double-clicked .cmd runs in the
rem     console default code page. Do not add "chcp" to this file either, since
rem     switching the page mid-file desynchronises the parser the same way.
setlocal
set "ROOT=%~dp0"

echo ==== WhisperTranscribe セットアップ ====
echo.

rem Prefer the py launcher pinned to 3.12, then any py, then python on PATH.
rem Each test needs its own if-block: "if cond cmd && cmd2" would run cmd2 off a
rem stale errorlevel when the condition is false.
set "PY="
py -3.12 -V >nul 2>&1 && set "PY=py -3.12"
if not defined PY (
    py -V >nul 2>&1 && set "PY=py"
)
if not defined PY (
    python -V >nul 2>&1 && set "PY=python"
)

if not defined PY (
    echo [エラー] Python が見つかりません。
    echo.
    echo   https://www.python.org/downloads/release/python-31210/
    echo   を開き、ページ下部の Windows installer 64-bit 版を入れてください。
    echo   インストーラ最初の画面の「Add python.exe to PATH」に
    echo   必ずチェックを入れます。
    echo.
    echo   インストール後、この セットアップ.cmd をもう一度実行してください。
    echo.
    pause
    exit /b 1
)

echo 使う Python:
%PY% -V
echo.

if exist "%ROOT%venv\Scripts\python.exe" (
    echo 専用の Python 環境はすでにあります。ライブラリの確認だけ行います。
) else (
    echo 専用の Python 環境を作成中...
    %PY% -m venv "%ROOT%venv"
    if errorlevel 1 goto failed
)
echo.

echo ライブラリをインストール中... 数分かかります。
"%ROOT%venv\Scripts\python.exe" -m pip install --upgrade pip
"%ROOT%venv\Scripts\python.exe" -m pip install -r "%ROOT%requirements.txt"
if errorlevel 1 goto failed

echo.
echo ==== 完了 ====
echo.
echo 「文字起こし開始.cmd」をダブルクリックすると使えます。
echo 初回だけモデル約 2.9GB のダウンロードが走るので、5分から15分ほどかかります。
echo 2 回目以降は数秒で起動します。
echo.
pause
exit /b 0

:failed
echo.
echo [エラー] セットアップに失敗しました。
echo 上に出ているメッセージを確認してください。
echo よくある原因: ネットワーク、社内プロキシ、ディスク空き容量不足。
echo.
pause
exit /b 1
