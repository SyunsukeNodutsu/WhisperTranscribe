#!/usr/bin/env python3
"""
PC で鳴っている音声を Whisper (faster-whisper) でリアルタイム文字起こしする。

・WASAPI ループバックで既定の再生デバイスの音を取得（仮想オーディオドライバ不要）
・--app を指定すると、そのアプリの音だけを文字起こしする（他アプリの音は入らない）
・無音を検出して区切り、区切りごとに GPU で認識してファイルへ追記
・Ctrl+C で停止。認識できた分は都度ファイルに書き込まれる
"""
import argparse
import configparser
import os
import queue
import site
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from math import gcd


def _add_cuda_dll_dirs():
    """pip で入れた cuBLAS / cuDNN の DLL を見つけられるようにする。

    CTranslate2 は実行時に名前で LoadLibrary するため、add_dll_directory だけでは
    見つけられない（cublas64_12.dll not found になる）。PATH にも通す必要がある。
    """
    dirs = []
    for pkg in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_nvrtc/bin"):
        for sp in site.getsitepackages():
            path = os.path.join(sp, *pkg.split("/"))
            if os.path.isdir(path):
                dirs.append(path)
                try:
                    os.add_dll_directory(path)
                except OSError:
                    pass
    if dirs:
        os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")


def _default_hf_home():
    """モデルの置き場をこのフォルダ内の models\\ にする。

    .cmd を経由せず直接実行された場合の保険（既定では C:\\Users\\...\\.cache に
    2.9GB 置かれてしまうため）。既に HF_HOME が設定されていればそれを尊重する。
    """
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


_add_cuda_dll_dirs()
_default_hf_home()

# Avast の HTTPS スキャンが証明書を差し替えるため、Python にも Windows の
# 証明書ストアを使わせる（モデル取得時の SSL エラー対策。検証は有効のまま）
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass

import numpy as np                       # noqa: E402
import pyaudiowpatch as pyaudio          # noqa: E402
from scipy.signal import resample_poly   # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

import proc_loopback                     # noqa: E402

TARGET_SR = 16000     # Whisper の入力サンプリングレート
BLOCK_MS = 100        # 音声を読む単位
MIN_SPEECH_MS = 400   # これ未満の発話は認識に回さない
IDLE_FLUSH_S = 1.5    # データが来なくなってこの秒数で区切る（再生停止時の対策）

# 無音区間で Whisper が創作しがちな定型句（音が小さい区間のみ除去する）
JUNK_PHRASES = (
    "ご視聴ありがとうございました",
    "ご覧いただきありがとうございました",
    "チャンネル登録",
    "最後までご視聴",
    "Thank you for watching",
    "Subtitles by",
    "字幕視聴",
)


def find_loopback(p, name_hint=None):
    """既定の再生デバイス（または名前指定）に対応するループバック入力を返す"""
    api = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    target = name_hint or p.get_device_info_by_index(api["defaultOutputDevice"])["name"]
    fallback = None
    for lb in p.get_loopback_device_info_generator():
        if fallback is None:
            fallback = lb
        if target in lb["name"]:
            return lb
    return fallback


def find_microphone(p, name_hint=None):
    api = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    if name_hint:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0 and name_hint in info["name"]:
                return info
    return p.get_device_info_by_index(api["defaultInputDevice"])


class Source:
    """1 つの入力を読み、無音で区切ってキューへ流す。

    「何から読むか」は open_stream に任せてあるので、再生デバイス全体でも
    特定プロセスの音でも同じ区切り処理を通せる。
    """

    def __init__(self, tag, q, args, stats, rate, channels, open_stream, label):
        self.tag, self.q, self.args, self.stats = tag, q, args, stats
        self.open_stream, self.label = open_stream, label
        self.rate = rate
        self.channels = channels
        self.block = int(self.rate * BLOCK_MS / 1000)
        g = gcd(self.rate, TARGET_SR)
        self.up, self.down = TARGET_SR // g, self.rate // g

        self.lock = threading.Lock()
        self.pre = deque(maxlen=3)     # 語頭を切らないための先読み (300ms)
        self.buf = []
        self.speech_ms = 0
        self.silence_ms = 0
        self.seg_start = None
        self.noise = 0.0
        self.last_data = time.time()

    # --- 呼び出し側は必ず lock を取ってから ---
    def _emit(self, forced=False):
        carry = []
        if self.buf and self.speech_ms >= MIN_SPEECH_MS:
            if forced and len(self.buf) > 12:
                # 時間切れの区切りは、末尾 1 秒のうち最も静かな所を選んで
                # 単語の途中で切れるのを避ける。残りは次のセグメントへ回す
                tail = self.buf[-10:]
                levels = [float(np.sqrt(np.mean(np.square(b)))) for b in tail]
                i = int(np.argmin(levels))
                if i < len(tail) - 1:
                    cut = len(self.buf) - 10 + i
                    carry, self.buf = self.buf[cut:], self.buf[:cut]
            self.q.put((self.tag, self.seg_start, np.concatenate(self.buf)))
            self.stats["queued"] += 1

        self.buf, self.speech_ms, self.silence_ms, self.seg_start = [], 0, 0, None
        self.pre.clear()
        if carry:
            self.buf = carry
            self.speech_ms = len(carry) * BLOCK_MS
            self.seg_start = datetime.now()

    def flush_if_idle(self):
        """再生が止まってデータが来なくなった場合の取り出し"""
        with self.lock:
            if self.buf and time.time() - self.last_data > IDLE_FLUSH_S:
                self._emit()

    def flush_final(self):
        with self.lock:
            self._emit()

    def run(self, stop):
        try:
            stream = self.open_stream()
        except Exception as e:
            print(f"\n[{self.tag}] 音声の取得を開始できませんでした: {e}", file=sys.stderr)
            stop.set()
            return

        max_ms = self.args.max_segment * 1000
        try:
            while not stop.is_set():
                try:
                    data = stream.read(self.block, exception_on_overflow=False)
                except Exception:
                    continue

                x = np.frombuffer(data, dtype=np.float32)
                if self.channels > 1:
                    x = x.reshape(-1, self.channels).mean(axis=1)
                if self.up != self.down:
                    x = resample_poly(x, self.up, self.down).astype(np.float32)
                if x.size == 0:
                    continue

                rms = float(np.sqrt(np.mean(np.square(x))))
                with self.lock:
                    self.last_data = time.time()
                    is_speech = rms > max(self.args.threshold, self.noise * 3.0)
                    if not is_speech:
                        # 暗騒音は「発話でない区間」だけで更新する。発話中も混ぜて
                        # 更新すると、講演のように喋りっぱなしの素材では推定値が
                        # 発話音量まで上がり、数分後から声を無音と判定して落とす
                        self.noise = 0.995 * self.noise + 0.005 * rms

                    if is_speech:
                        if not self.buf:
                            self.buf.extend(self.pre)      # 直前の 300ms も含める
                            self.seg_start = datetime.now()
                        self.buf.append(x)
                        self.speech_ms += BLOCK_MS
                        self.silence_ms = 0
                    elif self.buf:
                        self.buf.append(x)
                        self.silence_ms += BLOCK_MS
                    else:
                        self.pre.append(x)

                    long_enough = self.speech_ms >= MIN_SPEECH_MS
                    if self.buf and self.silence_ms >= self.args.min_silence and long_enough:
                        self._emit()
                    elif self.buf and self.speech_ms + self.silence_ms >= max_ms:
                        self._emit(forced=True)
        finally:
            # PyAudio と ProcessLoopbackStream で後始末の作法が違うので両方試す
            for name in ("stop_stream", "close"):
                try:
                    getattr(stream, name)()
                except Exception:
                    pass


def make_device_source(p, dev, tag, q, args, stats):
    """従来どおり、再生デバイス全体（またはマイク）から読む Source"""
    rate = int(dev["defaultSampleRate"])
    channels = min(int(dev["maxInputChannels"]), 2)
    block = int(rate * BLOCK_MS / 1000)

    def open_stream():
        return p.open(format=pyaudio.paFloat32, channels=channels, rate=rate,
                      input=True, input_device_index=dev["index"],
                      frames_per_buffer=block)

    return Source(tag, q, args, stats, rate, channels, open_stream, dev["name"])


def make_process_source(pid, label, tag, q, args, stats):
    """特定プロセスツリーの音だけから読む Source"""
    rate, channels = 48000, 2          # プロセスループバックは自分で形式を決める

    def open_stream():
        return proc_loopback.ProcessLoopbackStream(pid, rate=rate, channels=channels)

    return Source(tag, q, args, stats, rate, channels, open_stream, label)


def transcribe_worker(model, q, writer, args, stats, lock):
    while True:
        item = q.get()
        if item is None:
            break
        tag, ts, audio = item
        try:
            segments, _ = model.transcribe(
                audio,
                language=args.lang,
                beam_size=args.beam,
                vad_filter=True,
                condition_on_previous_text=False,   # 直前の文に引きずられた暴走を防ぐ
                initial_prompt=args.prompt or None,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                no_speech_threshold=0.6,
            )
            # Whisper 自身の文単位で 1 行にする（読みやすさと時刻精度のため）
            found = [(s.start, s.text.strip(), s.no_speech_prob, s.avg_logprob)
                     for s in segments if s.text.strip()]
        except Exception as e:
            print(f"\n認識エラー: {e}", file=sys.stderr)
            continue

        quiet = float(np.sqrt(np.mean(np.square(audio)))) < 0.01
        base = ts or datetime.now()
        for offset, text, no_speech, logprob in found:
            # 定型句の幻聴を捨てる。BGM があると音量では判別できないので、
            # Whisper 自身の「発話ではない確率」と自信度も見る
            if any(j in text for j in JUNK_PHRASES):
                if quiet or no_speech > 0.3 or logprob < -0.7:
                    continue
            parts = []
            if not args.no_timestamp:
                parts.append(f"[{base + timedelta(seconds=offset):%H:%M:%S}]")
            if args.mic:
                parts.append(f"[{tag}]")
            parts.append(text)
            line = " ".join(parts)
            with lock:
                writer.write(line + "\n")
                writer.flush()
                stats["lines"] += 1
                if args.echo:
                    print(line, flush=True)


def _to_bool(v):
    s = str(v).strip().lower()
    if s in ("true", "yes", "on", "1"):
        return True
    if s in ("false", "no", "off", "0"):
        return False
    raise ValueError("true か false で書いてください")


# settings.ini のキー -> (コマンドラインオプションの受け皿, 変換関数)
INI_SPEC = {
    "model":         ("model",        str),
    "language":      ("lang",         str),
    "device":        ("device",       str),
    "compute":       ("compute",      str),
    "beam":          ("beam",         int),
    "prompt":        ("prompt",       str),
    "mic":           ("mic",          _to_bool),
    "output_device": ("device_name",  str),
    "mic_device":    ("mic_name",     str),
    "app":           ("app",          str),
    "window":        ("app",          str),   # 旧称。古い settings.ini でも動くように残す
    "threshold":     ("threshold",    float),
    "min_silence":   ("min_silence",  int),
    "max_segment":   ("max_segment",  int),
    "echo":          ("echo",         _to_bool),
    "timestamp":     ("no_timestamp", lambda v: not _to_bool(v)),
    "out":           ("out",          str),
    "duration":      ("duration",     int),
}


def load_ini(path):
    """settings.ini を読み、argparse の既定値に流し込む dict と警告を返す。

    値が変でも落とさず、その項目だけ無視して警告する（手で編集する前提のため）。
    """
    if not os.path.exists(path):
        return {}, []
    cp = configparser.ConfigParser()
    try:
        cp.read(path, encoding="utf-8-sig")
    except Exception as e:
        return {}, [f"settings.ini を読めませんでした: {e}"]
    if not cp.has_section("settings"):
        return {}, ["settings.ini に [settings] の行が見つかりません"]

    values, warns = {}, []
    for key, raw in cp.items("settings"):
        if key not in INI_SPEC:
            warns.append(f"知らない設定名なので無視しました: {key}")
            continue
        raw = raw.strip()
        if raw == "":
            continue                       # 空欄は既定値のまま
        dest, convert = INI_SPEC[key]
        try:
            values[dest] = convert(raw)
        except Exception as e:
            warns.append(f"{key} の値 '{raw}' が不正なので無視しました（{e}）")
    return values, warns


def parse_args():
    ap = argparse.ArgumentParser(description="PC の音声を Whisper で文字起こしする")
    ap.add_argument("--out", help="出力ファイル（既定: transcripts\\YYYYmmdd-HHMMSS.txt）")
    ap.add_argument("--model", default="large-v3",
                    help="モデル名 large-v3 / large-v3-turbo / medium など（既定: large-v3）")
    ap.add_argument("--lang", default="ja", help="言語コード（既定: ja、自動判定は auto）")
    ap.add_argument("--compute", default="float16", help="演算精度（既定: float16）")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="推論デバイス")
    ap.add_argument("--beam", type=int, default=5, help="ビームサイズ（既定: 5）")
    ap.add_argument("--prompt", default="", help="固有名詞や専門用語を並べておくと認識が寄る")
    ap.add_argument("--mic", action="store_true",
                    help="システム音声に加えてマイクも文字起こしし [SYS]/[MIC] を付ける")
    ap.add_argument("--app",
                    help="このウィンドウタイトルを含むアプリの音だけを文字起こしする"
                         "（部分一致。拾うのはアプリ単位で、同じアプリの別タブの音も入る）")
    ap.add_argument("--device-name", help="ループバック元の再生デバイス名の一部を指定")
    ap.add_argument("--mic-name", help="使用するマイク名の一部を指定")
    ap.add_argument("--threshold", type=float, default=0.006,
                    help="発話とみなす音量の下限 RMS（既定: 0.006）")
    ap.add_argument("--min-silence", type=int, default=700,
                    help="この無音 ms で区切る（既定: 700）")
    ap.add_argument("--max-segment", type=int, default=25,
                    help="無音が来なくてもこの秒数で区切る（既定: 25）")
    ap.add_argument("--no-timestamp", action="store_true", help="行頭の時刻を付けない")
    ap.add_argument("--echo", action="store_true", help="認識結果を画面にも表示する")
    ap.add_argument("--duration", type=int, default=0,
                    help="指定秒数で自動終了（0 = Ctrl+C まで無制限）")
    ap.add_argument("--list-devices", action="store_true", help="デバイス一覧を表示して終了")
    ap.add_argument("--list-apps", action="store_true",
                    help="--app に指定できるアプリ一覧を表示して終了")

    # settings.ini を既定値として流し込む。コマンドラインで指定した分はそちらが勝つ
    ini_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.ini")
    ini_values, ini_warnings = load_ini(ini_path)
    if ini_values:
        ap.set_defaults(**ini_values)

    args = ap.parse_args()
    args.ini_loaded = bool(ini_values)
    args.ini_warnings = ini_warnings
    if args.lang == "auto":
        args.lang = None
    return args


def list_devices(p):
    api = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    print("既定の再生 :", p.get_device_info_by_index(api["defaultOutputDevice"])["name"])
    print("既定のマイク:", p.get_device_info_by_index(api["defaultInputDevice"])["name"])
    print("--- ループバック（--device-name で指定できます）---")
    for lb in p.get_loopback_device_info_generator():
        print(f"  {lb['name']}")
    print("--- マイク（--mic-name で指定できます）---")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0 and "[Loopback]" not in info["name"]:
            print(f"  {info['name']}")


def list_apps():
    """音を拾えるアプリを表示する。

    同じアプリのウィンドウはまとめて並べる。そうしないと「ウィンドウ単位で
    分けられる」と誤解しやすいため（実際に分けられるのはアプリ単位）。
    """
    table = proc_loopback.process_table()
    groups = {}
    for title, pid, exe in proc_loopback.list_windows():
        root = proc_loopback.resolve_audio_root(pid, table)
        group = groups.setdefault(root, {"exe": table.get(root, (0, exe))[1], "titles": []})
        group["titles"].append(title)

    print("--- 音を拾えるアプリ（--app にはウィンドウタイトルの一部を指定します）---")
    for root, info in sorted(groups.items(), key=lambda kv: kv[1]["exe"].lower()):
        print(f"  {info['exe']}  [PID {root}]")
        for title in sorted(info["titles"]):
            print(f"      {title}")
        if len(info["titles"]) > 1:
            print("      ※ 同じアプリなので、どれを指定しても上の音がまとめて入ります")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    args = parse_args()

    if args.list_apps:
        list_apps()
        return 0

    p = pyaudio.PyAudio()

    if args.list_devices:
        list_devices(p)
        p.terminate()
        return 0

    q = queue.Queue()
    stats = {"lines": 0, "queued": 0}
    sources = []

    if args.app:
        # 見つからなければ止める。黙って全体を拾うと「特定のアプリのつもりが
        # 全部拾っていた」という、記録し終わってから気づく事故になるため
        try:
            pid, title, window_pid, others = proc_loopback.find_app(args.app)
        except proc_loopback.ProcessLoopbackError as e:
            print(f"{e}", file=sys.stderr)
            p.terminate()
            return 1
        if others:
            # 別のアプリにも一致した場合は黙って選ばない。指定が曖昧なまま
            # 録り続けて、後から違うアプリだったと気づくのが一番困るため
            print(f"警告   : '{args.app}' は複数のアプリに一致しました。先頭を使います")
            for other_title, other_pid in others:
                print(f"         使わない候補: {other_title}  [PID {other_pid}]")
        exe = proc_loopback.process_table().get(pid, (0, "?"))[1]
        sources.append(make_process_source(pid, title, "SYS", q, args, stats))
        input_desc = f"アプリ: {exe}（{title}）"
        input_desc += (f"  [PID {window_pid} -> ツリー根 {pid}]" if window_pid != pid
                       else f"  [PID {pid}]")
    else:
        lb = find_loopback(p, args.device_name)
        if lb is None:
            print("ループバックデバイスが見つかりません。", file=sys.stderr)
            p.terminate()
            return 1
        sources.append(make_device_source(p, lb, "SYS", q, args, stats))
        input_desc = lb["name"]

    mic_desc = None
    if args.mic:
        mic = find_microphone(p, args.mic_name)
        sources.append(make_device_source(p, mic, "MIC", q, args, stats))
        mic_desc = mic["name"]

    if args.out:
        out_path = args.out
    else:
        out_dir = os.path.join(here, "transcripts")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt")

    for w in args.ini_warnings:
        print(f"設定の警告: {w}")
    print(f"設定     : {'settings.ini' if args.ini_loaded else '既定値（settings.ini なし）'}")
    print(f"音声入力 : {input_desc}")
    if mic_desc:
        print(f"マイク   : {mic_desc}")
    print(f"モデル   : {args.model} ({args.device}/{args.compute})")
    print(f"区切り   : 無音 {args.min_silence}ms / 最長 {args.max_segment}s")
    print("読み込み中...", end="", flush=True)
    t0 = time.time()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute)
    print(f" 完了 ({time.time() - t0:.1f}s)")
    print(f"保存先   : {out_path}")
    print("停止     : Ctrl+C")
    print()

    writer = open(out_path, "a", encoding="utf-8-sig")
    stop = threading.Event()
    write_lock = threading.Lock()

    worker = threading.Thread(target=transcribe_worker,
                             args=(model, q, writer, args, stats, write_lock), daemon=True)
    worker.start()
    for s in sources:
        threading.Thread(target=s.run, args=(stop,), daemon=True).start()

    deadline = time.time() + args.duration if args.duration else float("inf")
    try:
        while not stop.is_set() and time.time() < deadline:
            for s in sources:
                s.flush_if_idle()     # 再生が止まったまま溜まっている分を送る
            if not args.echo:
                print(f"\r保存した行数: {stats['lines']}   処理待ち: {q.qsize()}    ",
                      end="", flush=True)
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass

    stop.set()
    for s in sources:
        s.flush_final()               # 途切れた最後の発話も拾う
    print("\n残りを処理中...")
    q.put(None)
    worker.join(timeout=180)
    with write_lock:
        writer.close()
    print(f"終了。{stats['lines']} 行を保存しました -> {out_path}")
    sys.stdout.flush()
    # 録音スレッドは WASAPI の read で止まっていることがあり、
    # p.terminate() や通常終了だとハングするため強制終了する
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
