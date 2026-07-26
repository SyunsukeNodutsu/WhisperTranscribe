#!/usr/bin/env python3
"""特定のアプリが鳴らしている音だけを取り出す。

拾う単位は「アプリ（プロセスツリー）」であって、ウィンドウやタブではない。
Chrome のように 1 つのアプリが複数のウィンドウ・タブを持つ場合、それらの音は
まとめて入る。分けたい場合は別のアプリで開くしかない（Windows の API の制約）。


Windows のプロセスループバック API を ctypes だけで呼ぶ。仮想オーディオドライバも
追加パッケージも不要で、対象アプリの音は普段どおりスピーカーからも聞こえたまま。

transcribe.py 側からは PyAudio のストリームと同じ感覚で使える:

    stream = ProcessLoopbackStream(pid)
    data = stream.read(1600)      # float32 のバイト列が返る
    stream.close()

必要環境: Windows 10 ビルド 20348 以降（Windows 11 なら問題ない）
"""
import ctypes
import threading
import time
from ctypes import POINTER, byref, c_uint32, c_uint64, c_ulong, c_void_p, wintypes

import numpy as np

ole32 = ctypes.WinDLL("ole32")
user32 = ctypes.WinDLL("user32")
kernel32 = ctypes.WinDLL("kernel32")
mmdevapi = ctypes.WinDLL("Mmdevapi")


class ProcessLoopbackError(RuntimeError):
    """このモジュール由来の失敗。呼び出し側はこれだけ捕まえればよい"""


# ------------------------------------------------------------------ COM の土台

class GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]


def _guid(text):
    g = GUID()
    if ole32.IIDFromString(ctypes.c_wchar_p(text), byref(g)) != 0:
        raise ValueError(f"GUID が不正: {text}")
    return g


def _guid_bytes(g):
    return bytes(ctypes.cast(byref(g), POINTER(ctypes.c_byte * 16))[0])


IID_IAudioClient = _guid("{1CB9AD4C-DBFA-4C32-B178-C2F568A703B2}")
IID_IAudioCaptureClient = _guid("{C8ADBD64-E71E-48A0-A4DE-185C395CD317}")
IID_ICompletionHandler = _guid("{41D949AB-9862-444A-80F6-C261334DA5EB}")
IID_IUnknown = _guid("{00000000-0000-0000-C000-000000000046}")
IID_IMarshal = _guid("{00000003-0000-0000-C000-000000000046}")
# メソッドを持たないマーカーインターフェース。ActivateAudioInterfaceAsync は
# まずこれを QueryInterface してきて、E_NOINTERFACE を返すと
# E_ILLEGAL_METHOD_CALL で即失敗する。C++ の FtmBase が暗黙に満たしていた条件
IID_IAgileObject = _guid("{94EA2B94-E9CC-49E0-C0FF-EE64CA8F5B90}")

E_NOINTERFACE = -2147467262
VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK = "VAD\\Process_Loopback"

_HRESULT_HINTS = {
    0x80070057: "E_INVALIDARG（PID か指定内容が不正）",
    0x80070490: "ERROR_NOT_FOUND（対象プロセスが存在しない）",
    0x88890008: "AUDCLNT_E_UNSUPPORTED_FORMAT",
    0x8889000A: "AUDCLNT_E_DEVICE_IN_USE",
    0x88890004: "AUDCLNT_E_DEVICE_INVALIDATED",
    0x8000000E: "E_ILLEGAL_METHOD_CALL",
    0x80004001: "E_NOTIMPL（この Windows はプロセス単位キャプチャに非対応）",
}


def _check(hr, what):
    if hr < 0:
        code = hr & 0xFFFFFFFF
        hint = _HRESULT_HINTS.get(code, "")
        raise ProcessLoopbackError(f"{what} に失敗: 0x{code:08X} {hint}".rstrip())
    return hr


def _call(obj, index, restype, argtypes, *args):
    """COM の vtable の index 番目を呼ぶ。comtypes を使わずに済ませるための最小実装"""
    vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
    return ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)(vtbl[index])(obj, *args)


def _release(obj):
    if obj:
        _call(obj, 2, c_ulong, [])


# ------------------------------------------------- 完了ハンドラ（COM オブジェクト）

_QueryInterfaceFn = ctypes.WINFUNCTYPE(ctypes.c_long, c_void_p, POINTER(GUID),
                                       POINTER(c_void_p))
_AddRefFn = ctypes.WINFUNCTYPE(c_ulong, c_void_p)
_ReleaseFn = ctypes.WINFUNCTYPE(c_ulong, c_void_p)
_ActivateCompletedFn = ctypes.WINFUNCTYPE(ctypes.c_long, c_void_p, c_void_p)


class _HandlerVtbl(ctypes.Structure):
    _fields_ = [("QueryInterface", _QueryInterfaceFn), ("AddRef", _AddRefFn),
                ("Release", _ReleaseFn), ("ActivateCompleted", _ActivateCompletedFn)]


class _HandlerObj(ctypes.Structure):
    _fields_ = [("lpVtbl", POINTER(_HandlerVtbl))]


class _CompletionHandler:
    """アクティベーション完了を Event で知らせるだけの COM オブジェクト。

    Python 側に COM の実体が無いので vtable を手で組む。GC されると即クラッシュ
    するため、生成物は自分自身に持たせて寿命を呼び出し側と一致させる。
    """

    def __init__(self):
        self.done = threading.Event()
        self._ftm = c_void_p()

        def query_interface(this, riid, ppv):
            want = bytes(ctypes.cast(riid, POINTER(ctypes.c_byte * 16))[0])
            if want == _guid_bytes(IID_IMarshal) and self._ftm:
                return _call(self._ftm, 0, ctypes.c_long,
                             [POINTER(GUID), POINTER(c_void_p)], riid, ppv)
            for iid in (IID_IUnknown, IID_ICompletionHandler, IID_IAgileObject):
                if want == _guid_bytes(iid):
                    ppv[0] = this
                    return 0
            ppv[0] = None
            return E_NOINTERFACE

        def activate_completed(this, operation):
            self.done.set()          # 実際の結果は呼び出し側が取りに行く
            return 0

        self._vtbl = _HandlerVtbl(_QueryInterfaceFn(query_interface),
                                  _AddRefFn(lambda this: 2),
                                  _ReleaseFn(lambda this: 1),
                                  _ActivateCompletedFn(activate_completed))
        self._obj = _HandlerObj(ctypes.pointer(self._vtbl))
        self.pointer = ctypes.cast(ctypes.pointer(self._obj), c_void_p)

        ole32.CoCreateFreeThreadedMarshaler.restype = ctypes.c_long
        _check(ole32.CoCreateFreeThreadedMarshaler(self.pointer, byref(self._ftm)),
               "CoCreateFreeThreadedMarshaler")


# ------------------------------------------------------- アクティベーション用の型

AUDCLNT_SHAREMODE_SHARED = 0
AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
AUDCLNT_STREAMFLAGS_EVENTCALLBACK = 0x00040000
AUDCLNT_BUFFERFLAGS_SILENT = 0x2
AUDCLNT_E_UNSUPPORTED_FORMAT = 0x88890008
ACTIVATION_TYPE_PROCESS_LOOPBACK = 1
INCLUDE_TARGET_PROCESS_TREE = 0
VT_BLOB = 65
WAVE_FORMAT_PCM = 1
WAVE_FORMAT_IEEE_FLOAT = 3


class _LoopbackParams(ctypes.Structure):
    _fields_ = [("TargetProcessId", wintypes.DWORD), ("ProcessLoopbackMode", ctypes.c_int)]


class _ActivationParams(ctypes.Structure):
    _fields_ = [("ActivationType", ctypes.c_int), ("ProcessLoopbackParams", _LoopbackParams)]


class _BLOB(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.ULONG), ("pBlobData", POINTER(ctypes.c_byte))]


class _PROPVARIANT(ctypes.Structure):
    _fields_ = [("vt", wintypes.USHORT), ("wReserved1", wintypes.USHORT),
                ("wReserved2", wintypes.USHORT), ("wReserved3", wintypes.USHORT),
                ("blob", _BLOB)]


class WAVEFORMATEX(ctypes.Structure):
    _fields_ = [("wFormatTag", wintypes.WORD), ("nChannels", wintypes.WORD),
                ("nSamplesPerSec", wintypes.DWORD), ("nAvgBytesPerSec", wintypes.DWORD),
                ("nBlockAlign", wintypes.WORD), ("wBitsPerSample", wintypes.WORD),
                ("cbSize", wintypes.WORD)]


def _make_format(tag, rate, channels, bits):
    fmt = WAVEFORMATEX()
    fmt.wFormatTag = tag
    fmt.nChannels = channels
    fmt.nSamplesPerSec = rate
    fmt.wBitsPerSample = bits
    fmt.nBlockAlign = channels * bits // 8
    fmt.nAvgBytesPerSec = rate * fmt.nBlockAlign
    fmt.cbSize = 0
    return fmt


# ------------------------------------------------------------ プロセス／ウィンドウ

TH32CS_SNAPPROCESS = 0x2
MAX_PATH = 260


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", POINTER(c_ulong)),
                ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * MAX_PATH)]


def process_table():
    """{pid: (親pid, 実行ファイル名)}"""
    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap == -1:
        return {}
    table = {}
    entry = _PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
    ok = kernel32.Process32FirstW(snap, byref(entry))
    while ok:
        table[entry.th32ProcessID] = (entry.th32ParentProcessID, entry.szExeFile)
        ok = kernel32.Process32NextW(snap, byref(entry))
    kernel32.CloseHandle(snap)
    return table


def resolve_audio_root(pid, table=None):
    """音を出しているプロセスを含むツリーの根を返す。

    Chrome や Electron 系はウィンドウを持つプロセスと音を出すプロセスが別なので、
    ウィンドウから得た PID をそのまま指定すると何も録れない。「親が同じ実行ファイル名
    なら遡る」規則でツリーの根に寄せる（chrome.exe -> chrome.exe -> explorer.exe で停止）。
    """
    table = table or process_table()
    if pid not in table:
        return pid
    current = pid
    for _ in range(16):                       # 万一循環していても抜ける
        parent, name = table[current]
        if parent not in table:
            break
        if table[parent][1].lower() != name.lower():
            break
        current = parent
    return current


def list_windows():
    """[(タイトル, pid, 実行ファイル名)] を返す。タイトルのある可視ウィンドウのみ"""
    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    table = process_table()
    found = []

    def callback(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, byref(pid))
        found.append((buf.value, pid.value, table.get(pid.value, (0, "?"))[1]))
        return True

    user32.EnumWindows(WNDENUMPROC(callback), 0)
    return found


def find_app(name_hint):
    """ウィンドウタイトルの一部から、音を拾うアプリ（プロセスツリー）を決める。

    探す手がかりはウィンドウタイトルだが、拾えるのはアプリ単位である点に注意。
    同じアプリの別ウィンドウ・別タブの音はまとめて入る（Windows の API の制約で、
    ウィンドウやタブ単位で音を分ける方法は存在しない）。

    照合は部分一致・大文字小文字の区別なし。複数一致したときは先頭を採用し、
    別アプリの候補があれば others に入れて呼び出し側が知らせられるようにする。

    戻り値: (キャプチャ対象pid, 一致したタイトル, ウィンドウのpid, 別アプリの候補)
    """
    matches = [w for w in list_windows() if name_hint.lower() in w[0].lower()]
    if not matches:
        raise ProcessLoopbackError(
            f"'{name_hint}' を含むウィンドウが見つかりません（--list-apps で確認できます）")
    table = process_table()
    resolved = [(title, pid, resolve_audio_root(pid, table)) for title, pid, _ in matches]
    title, window_pid, root = resolved[0]
    others = [(t, r) for t, _, r in resolved[1:] if r != root]
    return root, title, window_pid, others


# ---------------------------------------------------------------- ストリーム本体

class ProcessLoopbackStream:
    """指定プロセスツリーの音を読み出す。PyAudio のストリームと同じ感覚で使える。

    読み出しは別スレッドで先回りして行い、read() は溜まった分から切り出す。
    対象が無音のときはデータが来ないことがあるため、その場合は無音を返して
    呼び出し側の処理が止まらないようにする（無音判定は呼び出し側に任せる）。
    """

    def __init__(self, pid, rate=48000, channels=2, buffer_seconds=4.0):
        self.pid = pid
        self.rate = rate
        self.channels = channels
        self.frame_bytes = channels * 4          # 呼び出し側には常に float32 で渡す
        self._max_bytes = int(rate * buffer_seconds) * self.frame_bytes

        self._buf = bytearray()
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error = None

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(15.0):
            self._stop.set()
            raise ProcessLoopbackError("キャプチャの開始が確認できませんでした")
        if self._error:
            raise self._error

    # ------------------------------------------------------------ 呼び出し側 API
    def read(self, nframes, exception_on_overflow=False):
        """nframes 分の float32 バイト列を返す（PyAudio の read と同じ形）"""
        need = nframes * self.frame_bytes
        deadline = time.time() + max(0.5, nframes / self.rate * 4)
        with self._cond:
            while len(self._buf) < need and not self._stop.is_set():
                if not self._cond.wait(timeout=max(0.05, deadline - time.time())):
                    if time.time() >= deadline:
                        break                    # 対象が無音。無音を返して進める
            if len(self._buf) >= need:
                data = bytes(self._buf[:need])
                del self._buf[:need]
                return data
            data = bytes(self._buf)
            self._buf.clear()
        return data + b"\x00" * (need - len(data))

    def close(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=3.0)

    # ------------------------------------------------------------- 内部スレッド
    def _activate(self):
        params = _ActivationParams()
        params.ActivationType = ACTIVATION_TYPE_PROCESS_LOOPBACK
        params.ProcessLoopbackParams.TargetProcessId = self.pid
        params.ProcessLoopbackParams.ProcessLoopbackMode = INCLUDE_TARGET_PROCESS_TREE

        prop = _PROPVARIANT()
        prop.vt = VT_BLOB
        prop.blob.cbSize = ctypes.sizeof(params)
        prop.blob.pBlobData = ctypes.cast(byref(params), POINTER(ctypes.c_byte))

        handler = _CompletionHandler()
        operation = c_void_p()
        mmdevapi.ActivateAudioInterfaceAsync.restype = ctypes.c_long
        _check(mmdevapi.ActivateAudioInterfaceAsync(
            ctypes.c_wchar_p(VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK),
            byref(IID_IAudioClient), byref(prop), handler.pointer, byref(operation)),
            "ActivateAudioInterfaceAsync")

        if not handler.done.wait(5.0):
            raise ProcessLoopbackError("アクティベーションの完了通知が来ませんでした")

        activate_hr = ctypes.c_long()
        client = c_void_p()
        _check(_call(operation, 3, ctypes.c_long,
                     [POINTER(ctypes.c_long), POINTER(c_void_p)],
                     byref(activate_hr), byref(client)), "GetActivateResult")
        _check(activate_hr.value, "音声インターフェースのアクティベーション")
        _release(operation)
        return client

    def _initialize(self, client):
        """float32 で初期化を試み、拒否されたら 16bit PCM に落とす。

        戻り値は「読み出したデータを float32 に変換する必要があるか」
        """
        attempts = [(WAVE_FORMAT_IEEE_FLOAT, 32, False), (WAVE_FORMAT_PCM, 16, True)]
        last = None
        for tag, bits, needs_convert in attempts:
            fmt = _make_format(tag, self.rate, self.channels, bits)
            hr = _call(client, 3, ctypes.c_long,
                       [ctypes.c_int, wintypes.DWORD, ctypes.c_longlong,
                        ctypes.c_longlong, POINTER(WAVEFORMATEX), c_void_p],
                       AUDCLNT_SHAREMODE_SHARED,
                       AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                       2000000, 0, byref(fmt), None)
            if hr >= 0:
                return needs_convert, fmt.nBlockAlign
            last = hr
            if (hr & 0xFFFFFFFF) != AUDCLNT_E_UNSUPPORTED_FORMAT:
                break
        _check(last, "IAudioClient::Initialize")

    def _run(self):
        client = capture = None
        event = None
        try:
            ole32.CoInitializeEx(None, 0)             # MTA
            client = self._activate()
            needs_convert, block_align = self._initialize(client)

            event = kernel32.CreateEventW(None, False, False, None)
            _check(_call(client, 13, ctypes.c_long, [wintypes.HANDLE], event),
                   "SetEventHandle")
            capture = c_void_p()
            _check(_call(client, 14, ctypes.c_long,
                         [POINTER(GUID), POINTER(c_void_p)],
                         byref(IID_IAudioCaptureClient), byref(capture)), "GetService")
            _check(_call(client, 10, ctypes.c_long, []), "IAudioClient::Start")
        except ProcessLoopbackError as e:
            self._error = e
            self._ready.set()
            return
        except Exception as e:                        # 想定外も呼び出し側へ返す
            self._error = ProcessLoopbackError(str(e))
            self._ready.set()
            return

        self._ready.set()
        try:
            while not self._stop.is_set():
                kernel32.WaitForSingleObject(event, 200)
                while not self._stop.is_set():
                    data = POINTER(ctypes.c_byte)()
                    nframes = c_uint32()
                    flags = wintypes.DWORD()
                    pos, qpc = c_uint64(), c_uint64()
                    hr = _call(capture, 3, ctypes.c_long,
                               [POINTER(POINTER(ctypes.c_byte)), POINTER(c_uint32),
                                POINTER(wintypes.DWORD), POINTER(c_uint64),
                                POINTER(c_uint64)],
                               byref(data), byref(nframes), byref(flags),
                               byref(pos), byref(qpc))
                    if hr < 0 or nframes.value == 0:
                        break
                    size = nframes.value * block_align
                    if flags.value & AUDCLNT_BUFFERFLAGS_SILENT:
                        chunk = b"\x00" * (nframes.value * self.frame_bytes)
                    else:
                        raw = ctypes.string_at(data, size)
                        if needs_convert:
                            samples = np.frombuffer(raw, dtype=np.int16)
                            chunk = (samples.astype(np.float32) / 32768.0).tobytes()
                        else:
                            chunk = raw
                    _call(capture, 4, ctypes.c_long, [c_uint32], nframes.value)
                    with self._cond:
                        self._buf.extend(chunk)
                        if len(self._buf) > self._max_bytes:   # 認識が詰まったら古い方を捨てる
                            del self._buf[:len(self._buf) - self._max_bytes]
                        self._cond.notify_all()
        finally:
            try:
                _call(client, 11, ctypes.c_long, [])           # Stop
            except Exception:
                pass
            if event:
                kernel32.CloseHandle(event)
            _release(capture)
            _release(client)
            with self._cond:
                self._cond.notify_all()
