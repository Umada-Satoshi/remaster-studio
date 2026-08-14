#!/usr/bin/env python3
"""
Audio Remaster Tool — 高音質 重低音重視 リマスター
=================================================
Usage:
    python3 remaster.py input.wav -o output.wav
    python3 remaster.py input.mp3 -o output.mp3 --bass-boost 8 --preset club
    python3 remaster.py input.wav --preset movie --analyze-only

Features:
    - 重低音ブースト (Low-Shelf EQ, 20-200Hz)
    - パラメトリックEQ (mid-range clarity)
    - マルチバンドコンプレッサー
    - リミッター (クリッピング防止)
    - LUFS正規化 (ITU-R BS.1770)
    - ステレオ拡張
    - FFTスペクトル解析
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq


# ── Presets ──────────────────────────────────────────────────────────────

PRESETS = {
    "default": {
        "bass_boost_db": 6.0,       # 重低音ブースト量 (dB)
        "bass_freq": 150.0,         # 重低音の境界周波数 (Hz)
        "sub_bass_freq": 60.0,      # サブベースの中心 (Hz)
        "sub_bass_boost_db": 4.0,   # サブベースブースト
        "mid_eq_freq": 3000.0,      # ミッドレンジEQ (Hz)
        "mid_eq_db": 2.0,           # ミッドブースト (クリアリティ)
        "high_shelf_freq": 8000.0,  # ハイシェルフ (Hz)
        "high_shelf_db": 1.5,       # 高域明るさ
        "compression_threshold": -20.0,  # コンプ閾値 (dBFS)
        "compression_ratio": 3.0,   # コンプレッサー比
        "limiter_ceiling": -0.5,    # リミッター上限 (dBFS)
        "target_lufs": -14.0,       # 目標ラウドネス
        "stereo_width": 1.2,        # ステレオ幅 (1.0=元のまま)
        "sample_rate": 48000,       # 出力サンプルレート
    },
    "club": {
        "bass_boost_db": 10.0,
        "bass_freq": 150.0,
        "sub_bass_freq": 50.0,
        "sub_bass_boost_db": 8.0,
        "mid_eq_freq": 2500.0,
        "mid_eq_db": 1.0,
        "high_shelf_freq": 8000.0,
        "high_shelf_db": 0.5,
        "compression_threshold": -18.0,
        "compression_ratio": 4.0,
        "limiter_ceiling": -0.3,
        "target_lufs": -10.0,
        "stereo_width": 1.3,
        "sample_rate": 48000,
    },
    "movie": {
        "bass_boost_db": 8.0,
        "bass_freq": 120.0,
        "sub_bass_freq": 45.0,
        "sub_bass_boost_db": 6.0,
        "mid_eq_freq": 2000.0,
        "mid_eq_db": 3.0,
        "high_shelf_freq": 10000.0,
        "high_shelf_db": 2.0,
        "compression_threshold": -24.0,
        "compression_ratio": 2.5,
        "limiter_ceiling": -1.0,
        "target_lufs": -24.0,       # ムービーは標準-24 LUFS
        "stereo_width": 1.4,
        "sample_rate": 48000,
    },
    "audiophile": {
        "bass_boost_db": 4.0,
        "bass_freq": 150.0,
        "sub_bass_freq": 60.0,
        "sub_bass_boost_db": 3.0,
        "mid_eq_freq": 3000.0,
        "mid_eq_db": 1.0,
        "high_shelf_freq": 8000.0,
        "high_shelf_db": 1.0,
        "compression_threshold": -22.0,
        "compression_ratio": 2.0,
        "limiter_ceiling": -0.5,
        "target_lufs": -16.0,
        "stereo_width": 1.1,
        "sample_rate": 96000,       # 高解像度
    },
}


# ── Audio I/O (ffmpeg based) ─────────────────────────────────────────────

def read_audio(path: str, target_sr: int = 48000) -> tuple[np.ndarray, int]:
    """ffmpegで任意形式→wavに変換してnumpy配列で返す"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(path),
            "-ar", str(target_sr),
            "-ac", "2",
            "-sample_fmt", "s16",
            tmp_wav
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        with wave.open(tmp_wav, "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        # stereo: (N, 2) → (2, N)
        if n_channels == 2:
            audio = audio.reshape(-1, 2).T
        else:
            audio = audio.reshape(1, -1)

        # normalize to [-1, 1]
        audio = audio / 32768.0
        return audio, sr
    finally:
        os.unlink(tmp_wav)


def write_audio(audio: np.ndarray, sr: int, path: str):
    """numpy配列→ffmpegで任意形式に書き出し"""
    # clamp
    audio = np.clip(audio, -1.0, 1.0)
    # float→int16
    audio_int = (audio * 32767).astype(np.int16)
    # (2, N) → (N, 2)
    if audio_int.shape[0] == 2:
        audio_int = audio_int.T

    ext = Path(path).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        with wave.open(tmp_wav, "wb") as wf:
            wf.setnchannels(audio_int.shape[1])
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio_int.tobytes())

        # convert to target format
        cmd = ["ffmpeg", "-y", "-i", tmp_wav]
        if ext == ".mp3":
            cmd += ["-codec:a", "libmp3lame", "-b:a", "320k"]
        elif ext == ".flac":
            cmd += ["-codec:a", "flac"]
        elif ext == ".ogg":
            cmd += ["-codec:a", "libvorbis", "-q:a", "10"]
        elif ext == ".wav":
            cmd += ["-codec:a", "pcm_s16le"]
        else:
            cmd += ["-codec:a", "pcm_s16le"]
        cmd.append(str(path))
        subprocess.run(cmd, capture_output=True, check=True)
    finally:
        os.unlink(tmp_wav)


# ── DSP Processing ───────────────────────────────────────────────────────

def design_biquad_filter(filter_type: str, freq: float, sr: int,
                         gain_db: float = 0.0, q: float = 0.707) -> np.ndarray:
    """IIR biquadフィルタのb, a係数を返す"""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)

    if filter_type == "low_shelf":
        b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
    elif filter_type == "high_shelf":
        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
    elif filter_type == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def apply_iir(audio: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """IIRフィルタを各チャンネルに適用"""
    out = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        out[ch] = signal.lfilter(b, a, audio[ch])
    return out


def multiband_compress(audio: np.ndarray, sr: int,
                       thresholds: list = None,
                       ratios: list = None,
                       attack_ms: float = 10.0,
                       release_ms: float = 100.0) -> np.ndarray:
    """マルチバンドコンプレッサー (3バンド: low/mid/high)"""
    if thresholds is None:
        thresholds = [-20.0, -22.0, -24.0]
    if ratios is None:
        ratios = [3.0, 2.5, 2.0]

    # crossover frequencies
    low_mid = 200.0
    mid_high = 2000.0

    # bandpassフィルタでバンド分離
    bands = []
    for ch in range(audio.shape[0]):
        # Low band: 0 ~ 200Hz
        b_l, a_l = signal.butter(4, low_mid / (sr / 2), btype="low")
        low = signal.lfilter(b_l, a_l, audio[ch])

        # Mid band: 200Hz ~ 2kHz
        b_hp, a_hp = signal.butter(4, low_mid / (sr / 2), btype="high")
        b_lp, a_lp = signal.butter(4, mid_high / (sr / 2), btype="low")
        mid = signal.lfilter(b_hp, a_hp, audio[ch])
        mid = signal.lfilter(b_lp, a_lp, mid)

        # High band: 2kHz ~ Nyquist
        b_h, a_h = signal.butter(4, mid_high / (sr / 2), btype="high")
        high = signal.lfilter(b_h, a_h, audio[ch])

        bands.append((low, mid, high))

    # 各バンドにコンプレッサー適用
    attack_coeff = np.exp(-1.0 / (sr * attack_ms / 1000.0))
    release_coeff = np.exp(-1.0 / (sr * release_ms / 1000.0))

    compressed = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        low, mid, high = bands[ch]
        result = np.zeros(audio.shape[1])

        for band_data, thresh_db, ratio in zip(
            [low, mid, high], thresholds, ratios
        ):
            # envelope detection (RMS with window)
            window_size = int(sr * 0.02)  # 20ms window
            env = np.zeros(len(band_data))
            for i in range(0, len(band_data), window_size):
                chunk = band_data[i:i + window_size]
                rms = np.sqrt(np.mean(chunk ** 2) + 1e-10)
                env_db = 20 * np.log10(rms + 1e-10)
                env[i:i + window_size] = env_db

            # gain reduction
            gain = np.zeros(len(band_data))
            current_env = -100.0
            for i in range(len(band_data)):
                if env[i] > current_env:
                    current_env = attack_coeff * current_env + (1 - attack_coeff) * env[i]
                else:
                    current_env = release_coeff * current_env + (1 - release_coeff) * env[i]

                if current_env > thresh_db:
                    overshoot = current_env - thresh_db
                    reduction = overshoot * (1 - 1.0 / ratio)
                    gain[i] = -reduction
                else:
                    gain[i] = 0.0

            # apply gain (convert dB to linear)
            gain_linear = 10 ** (gain / 20.0)
            result += band_data * gain_linear

        compressed[ch] = result

    return compressed


def limiter(audio: np.ndarray, ceiling_db: float = -0.5,
            release_ms: float = 50.0) -> np.ndarray:
    """ Brick-wall リミッター (クリッピング防止) """
    ceiling = 10 ** (ceiling_db / 20.0)
    release_coeff = np.exp(-1.0 / (48000 * release_ms / 1000.0))

    out = np.copy(audio)
    for ch in range(out.shape[0]):
        gain = 1.0
        for i in range(out.shape[1]):
            abs_val = abs(out[ch, i])
            if abs_val * gain > ceiling:
                target_gain = ceiling / (abs_val + 1e-10)
                gain = min(gain, target_gain)
            else:
                gain = min(1.0, gain * release_coeff + (1 - release_coeff))
            out[ch, i] *= gain

    return out


def normalize_lufs(audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """LUFS正規化 (simplified ITU-R BS.1770)"""
    # K-weighting (simplified: high-pass + shelf)
    # Pre-filter: high-pass 38Hz
    b_hp, a_hp = signal.butter(2, 38.0 / (sr / 2), btype="high")
    # Relative equal-loudness: +4dB at 2kHz
    b_shelf, a_shelf = design_biquad_filter("high_shelf", 2500.0, sr, gain_db=4.0, q=0.707)

    weighted = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        w = signal.lfilter(b_hp, a_hp, audio[ch])
        w = signal.lfilter(b_shelf, a_shelf, w)
        weighted[ch] = w

    # Integrated LUFS (mean square of weighted signal)
    # Gate: -10 LU gate (simplified)
    gate_threshold = 10 ** ((-70.0) / 10.0)  # -70 LUFS gating
    sum_sq = 0.0
    count = 0
    for ch in range(weighted.shape[0]):
        for i in range(0, weighted.shape[1], sr):  # 1秒ブロック
            block = weighted[ch, i:i + sr]
            if len(block) > 0:
                ms = np.mean(block ** 2)
                if ms > gate_threshold:
                    sum_sq += ms
                    count += 1

    if count == 0:
        return audio

    lufs = -0.691 + 10 * np.log10(sum_sq / count + 1e-10)

    # gain to reach target LUFS
    gain_db = target_lufs - lufs
    gain = 10 ** (gain_db / 20.0)
    return audio * gain


def stereo_enhance(audio: np.ndarray, width: float = 1.2) -> np.ndarray:
    """ステレオ幅拡張 (M/S処理)"""
    if audio.shape[0] != 2:
        return audio

    mid = (audio[0] + audio[1]) / 2.0
    side = (audio[0] - audio[1]) / 2.0

    # sideを拡大
    side *= width

    left = mid + side
    right = mid - side

    return np.array([left, right])


def analyze_spectrum(audio: np.ndarray, sr: int) -> dict:
    """FFTスペクトル分析"""
    # monoに変換
    if audio.shape[0] == 2:
        mono = (audio[0] + audio[1]) / 2.0
    else:
        mono = audio[0]

    # フレーム分割
    nperseg = min(len(mono), sr * 4)  # 4秒ウィンドウ
    f, t, Zxx = signal.stft(mono, fs=sr, nperseg=nperseg)

    # パワースペクトル
    power = np.mean(np.abs(Zxx) ** 2, axis=1)
    power_db = 10 * np.log10(power + 1e-10)

    # バンド別エネルギー
    def band_energy(f_lo, f_hi):
        mask = (f >= f_lo) & (f < f_hi)
        if mask.any():
            return float(10 * np.log10(np.mean(power[mask]) + 1e-10))
        return -100.0

    bands = {
        "sub_bass (20-60Hz)": band_energy(20, 60),
        "bass (60-200Hz)": band_energy(60, 200),
        "low_mid (200-500Hz)": band_energy(200, 500),
        "mid (500-2kHz)": band_energy(500, 2000),
        "upper_mid (2-6kHz)": band_energy(2000, 6000),
        "high (6-12kHz)": band_energy(6000, 12000),
        "air (12-20kHz)": band_energy(12000, 20000),
    }

    # RMS
    rms = float(np.sqrt(np.mean(mono ** 2)))
    rms_db = float(20 * np.log10(rms + 1e-10))

    # Peak
    peak = float(np.max(np.abs(mono)))
    peak_db = float(20 * np.log10(peak + 1e-10))

    # Crest factor (dynamic range indicator)
    crest_db = peak_db - rms_db

    # DC offset
    dc_offset = float(np.mean(mono))

    # 静的クリッピング検出
    clipping_ratio = float(np.mean(np.abs(mono) > 0.99))

    return {
        "sample_rate": sr,
        "duration_sec": len(mono) / sr,
        "channels": "stereo" if audio.shape[0] == 2 else "mono",
        "rms_db": round(rms_db, 2),
        "peak_db": round(peak_db, 2),
        "crest_factor_db": round(crest_db, 2),
        "dc_offset": round(dc_offset, 6),
        "clipping_ratio": round(clipping_ratio, 6),
        "band_energy": bands,
    }


# ── Main Remaster Pipeline ───────────────────────────────────────────────

def remaster(audio: np.ndarray, sr: int, params: dict) -> np.ndarray:
    """メインリマスタリングパイプライン"""
    print(f"  [1/7] Low-Shelf EQ: +{params['bass_boost_db']}dB @ {params['bass_freq']}Hz...")
    b, a = design_biquad_filter("low_shelf", params["bass_freq"], sr,
                                 gain_db=params["bass_boost_db"], q=0.707)
    audio = apply_iir(audio, b, a)

    print(f"  [2/7] Sub-Bass EQ: +{params['sub_bass_boost_db']}dB @ {params['sub_bass_freq']}Hz...")
    b, a = design_biquad_filter("peaking", params["sub_bass_freq"], sr,
                                 gain_db=params["sub_bass_boost_db"], q=0.5)
    audio = apply_iir(audio, b, a)

    print(f"  [3/7] Mid EQ: +{params['mid_eq_db']}dB @ {params['mid_eq_freq']}Hz...")
    b, a = design_biquad_filter("peaking", params["mid_eq_freq"], sr,
                                 gain_db=params["mid_eq_db"], q=1.0)
    audio = apply_iir(audio, b, a)

    print(f"  [4/7] High-Shelf EQ: +{params['high_shelf_db']}dB @ {params['high_shelf_freq']}Hz...")
    b, a = design_biquad_filter("high_shelf", params["high_shelf_freq"], sr,
                                 gain_db=params["high_shelf_db"], q=0.707)
    audio = apply_iir(audio, b, a)

    print(f"  [5/7] Multiband Compressor (threshold={params['compression_threshold']}dB, ratio={params['compression_ratio']}:1)...")
    audio = multiband_compress(audio, sr,
                               thresholds=[params["compression_threshold"]] * 3,
                               ratios=[params["compression_ratio"]] * 3)

    print(f"  [6/7] Limiter (ceiling={params['limiter_ceiling']}dBFS)...")
    audio = limiter(audio, ceiling_db=params["limiter_ceiling"])

    print(f"  [7/7] LUFS Normalization (target={params['target_lufs']} LUFS)...")
    audio = normalize_lufs(audio, sr, params["target_lufs"])

    # ステレオ拡張
    if params["stereo_width"] != 1.0:
        print(f"  [+] Stereo Width: {params['stereo_width']}x...")
        audio = stereo_enhance(audio, params["stereo_width"])

    # 最終リミッター (安全確認)
    audio = limiter(audio, ceiling_db=-0.3)

    return np.clip(audio, -1.0, 1.0)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="高音質 重低音重視 リマスターツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Preset Examples:
  %(prog)s song.wav -o remastered.wav                    # デフォルト
  %(prog)s song.mp3 -o club.mp3 --preset club            # クラブ用
  %(prog)s movie.wav -o movie_remastered.wav --preset movie
  %(prog)s song.flac -o hifi.flac --preset audiophile
  %(prog)s song.wav --analyze-only                        # 分析のみ
        """
    )
    parser.add_argument("input", help="入力ファイル (wav/mp3/flac/ogg)")
    parser.add_argument("-o", "--output", help="出力ファイル", default=None)
    parser.add_argument("--preset", choices=list(PRESETS.keys()),
                        default="default", help="プリセット (default)")
    parser.add_argument("--bass-boost", type=float, default=None,
                        help="重低音ブースト量 (dB) — プリセットを上書き")
    parser.add_argument("--sub-bass-boost", type=float, default=None,
                        help="サブベースブースト (dB)")
    parser.add_argument("--mid-boost", type=float, default=None,
                        help="ミッドレンジブースト (dB)")
    parser.add_argument("--high-boost", type=float, default=None,
                        help="高域ブースト (dB)")
    parser.add_argument("--stereo-width", type=float, default=None,
                        help="ステレオ幅 (1.0=元のまま, 2.0=2倍)")
    parser.add_argument("--target-lufs", type=float, default=None,
                        help="目標ラウドネス (LUFS)")
    parser.add_argument("--sample-rate", type=int, default=None,
                        help="出力サンプルレート")
    parser.add_argument("--analyze-only", action="store_true",
                        help="分析のみ（リマスターしない）")
    parser.add_argument("--json", action="store_true",
                        help="分析結果をJSONで出力")
    parser.add_argument("--compare", action="store_true",
                        help="before/after比較を表示")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="詳細出力")

    args = parser.parse_args()

    # プリセット読み込み
    params = PRESETS[args.preset].copy()

    # CLIオーバーライド
    if args.bass_boost is not None:
        params["bass_boost_db"] = args.bass_boost
    if args.sub_bass_boost is not None:
        params["sub_bass_boost_db"] = args.sub_bass_boost
    if args.mid_boost is not None:
        params["mid_eq_db"] = args.mid_boost
    if args.high_boost is not None:
        params["high_shelf_db"] = args.high_boost
    if args.stereo_width is not None:
        params["stereo_width"] = args.stereo_width
    if args.target_lufs is not None:
        params["target_lufs"] = args.target_lufs
    if args.sample_rate is not None:
        params["sample_rate"] = args.sample_rate

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"エラー: ファイルが見つかりません: {input_path}", file=sys.stderr)
        sys.exit(1)

    # 読み込み
    print(f"📂 読み込み: {input_path}")
    audio, sr = read_audio(str(input_path), params["sample_rate"])
    print(f"   {audio.shape[0]}ch, {sr}Hz, {audio.shape[1]/sr:.1f}秒")

    # 分析
    print(f"\n📊 Before 分析:")
    before = analyze_spectrum(audio, sr)
    if args.json:
        print(json.dumps(before, indent=2, ensure_ascii=False))
    else:
        print(f"   RMS: {before['rms_db']} dBFS")
        print(f"   Peak: {before['peak_db']} dBFS")
        print(f"   Crest Factor: {before['crest_factor_db']} dB")
        print(f"   DC Offset: {before['dc_offset']}")
        print(f"   Clipping: {before['clipping_ratio']*100:.4f}%")
        print(f"   バンド別エネルギー:")
        for band, val in before["band_energy"].items():
            bar = "█" * max(0, int((val + 60) / 3))
            print(f"     {band:>22s}: {val:6.1f} dB {bar}")

    if args.analyze_only:
        sys.exit(0)

    # リマスター
    print(f"\n🔧 リマスター中 (preset={args.preset})...")
    remastered = remaster(audio, sr, params)

    # After分析
    print(f"\n📊 After 分析:")
    after = analyze_spectrum(remastered, sr)
    print(f"   RMS: {after['rms_db']} dBFS")
    print(f"   Peak: {after['peak_db']} dBFS")
    print(f"   Crest Factor: {after['crest_factor_db']} dB")
    for band, val in after["band_energy"].items():
        before_val = before["band_energy"][band]
        diff = val - before_val
        bar = "█" * max(0, int((val + 60) / 3))
        arrow = "↑" if diff > 0.5 else ("↓" if diff < -0.5 else "→")
        print(f"     {band:>22s}: {val:6.1f} dB ({arrow}{abs(diff):.1f}dB) {bar}")

    # 差分サマリー
    print(f"\n📈 変化サマリー:")
    print(f"   RMS変化: {after['rms_db'] - before['rms_db']:+.1f} dB")
    print(f"   低域変化: {after['band_energy']['bass (60-200Hz)'] - before['band_energy']['bass (60-200Hz)']:+.1f} dB")
    print(f"   サブベース変化: {after['band_energy']['sub_bass (20-60Hz)'] - before['band_energy']['sub_bass (20-60Hz)']:+.1f} dB")

    # 書き出し
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = input_path.parent / f"{input_path.stem}_remastered{input_path.suffix}"

    print(f"\n💾 書き出し: {out_path}")
    write_audio(remastered, sr, str(out_path))

    file_size = out_path.stat().st_size
    if file_size > 1024 * 1024:
        print(f"   {file_size / 1024 / 1024:.1f} MB")
    else:
        print(f"   {file_size / 1024:.1f} KB")

    print(f"\n✅ 完了！ nya~ (=^･ω･^=)")


if __name__ == "__main__":
    main()
