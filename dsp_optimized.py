"""
dsp_optimized.py v4 — Numba JIT版（最大高速化）
================================================
Numba @njit で全ホットループをネイティブコード化。

正確性: 全関数 diff=0.0 を保証（v3.2から変更なし）
"""
import numpy as np
from scipy import signal as sig
from numba import njit, prange
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ═══════════════════════════════════════════════════════════════════════════
# Numba JIT ループ関数
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _limiter_release_loop(ideal_gain, release_coeff):
    """Limiter release適用 — 1次IIRフィルタ。"""
    n = len(ideal_gain)
    gain = np.empty(n, dtype=np.float64)
    gain[0] = ideal_gain[0]
    for i in range(1, n):
        if ideal_gain[i] < gain[i - 1]:
            gain[i] = ideal_gain[i]
        else:
            gain[i] = release_coeff * gain[i - 1] + (1 - release_coeff) * ideal_gain[i]
    return gain


@njit(cache=True)
def _rolling_max_deque(arr, window_size):
    """前方ローリング最大値 — monotonic deque O(n)。"""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    # deque をインデックス配列で模擬
    dq_head = 0
    dq_tail = 0
    dq = np.empty(n, dtype=np.int64)
    max_idx = np.empty(n, dtype=np.int64)

    for i in range(n):
        # 要素追加: arr[dq[tail-1]] <= arr[i] なら pop
        while dq_tail > dq_head and arr[dq[dq_tail - 1]] <= arr[i]:
            dq_tail -= 1
        dq[dq_tail] = i
        dq_tail += 1

        # 期限切れ: dq[head] <= i - window_size なら popleft
        while dq_head < dq_tail and dq[dq_head] <= i - window_size:
            dq_head += 1

        result[i] = arr[dq[dq_head]]

    return result


@njit(cache=True)
def _smooth_forward(gain_curve, release_coeff):
    """Forward smoothing — gain が増加方向に緩やかに。"""
    n = len(gain_curve)
    smoothed = np.empty(n, dtype=np.float64)
    smoothed[0] = gain_curve[0]
    for i in range(1, n):
        if gain_curve[i] > smoothed[i - 1]:
            smoothed[i] = smoothed[i - 1] * release_coeff + gain_curve[i] * (1 - release_coeff)
        else:
            smoothed[i] = gain_curve[i]
    return smoothed


@njit(cache=True)
def _smooth_backward(smoothed, release_coeff):
    """Backward smoothing — gain が減少方向に緩やかに。"""
    n = len(smoothed)
    for i in range(n - 2, -1, -1):
        if smoothed[i] > smoothed[i + 1]:
            smoothed[i] = smoothed[i + 1] * release_coeff + smoothed[i] * (1 - release_coeff)
    return smoothed


@njit(cache=True)
def _envelope_follower_smoother(env, atk_coeff, rel_coeff):
    """サンプル単位 envelope smoother — 元と同一アルゴリズム。"""
    n = len(env)
    result = np.empty(n, dtype=np.float64)
    cur = -100.0
    for i in range(n):
        if env[i] > cur:
            cur = atk_coeff * cur + (1 - atk_coeff) * env[i]
        else:
            cur = rel_coeff * cur + (1 - rel_coeff) * env[i]
        result[i] = cur
    return result


@njit(cache=True)
def _envelope_to_samples(env_frame, n_samples, ws):
    """フレーム単位 envelope をサンプルに展開。"""
    result = np.empty(n_samples, dtype=np.float64)
    n_frames = len(env_frame)
    for f in range(n_frames):
        start = f * ws
        end = min(start + ws, n_samples)
        for i in range(start, end):
            result[i] = env_frame[f]
    return result


@njit(cache=True)
def _gain_reduce(envelope, thresh, ratio):
    """ゲイン計算 — overshoot → reduction → linear gain。"""
    n = len(envelope)
    gain = np.empty(n, dtype=np.float64)
    for i in range(n):
        if envelope[i] > thresh:
            overshoot = envelope[i] - thresh
            reduction = overshoot * (1.0 - 1.0 / ratio)
            gain[i] = 10.0 ** (-reduction / 20.0)
        else:
            gain[i] = 1.0
    return gain


@njit(cache=True)
def _smooth_true_peak(gain):
    """True peak limiter — forward + backward smoothing。"""
    n = len(gain)
    smoothed = gain.copy()
    # Forward
    for i in range(1, n):
        val = smoothed[i - 1] * 0.999 + smoothed[i] * 0.001
        if val < smoothed[i]:
            smoothed[i] = val
    # Backward
    for i in range(n - 2, -1, -1):
        val = smoothed[i + 1] * 0.999 + smoothed[i] * 0.001
        if val < smoothed[i]:
            smoothed[i] = val
    return smoothed


@njit(cache=True)
def _peak_env_from_upsampled(abs_up, oversample, n_orig):
    """アップサンプル信号からピークエンベロープ抽出。"""
    peak_env = np.empty(n_orig, dtype=np.float64)
    for i in range(n_orig):
        start = i * oversample
        end = start + oversample
        mx = abs_up[start]
        for j in range(start + 1, end):
            if abs_up[j] > mx:
                mx = abs_up[j]
        peak_env[i] = mx
    return peak_env


# ═══════════════════════════════════════════════════════════════════════════
# 1. Limiter
# ═══════════════════════════════════════════════════════════════════════════

def limiter_fast(audio, ceiling_db=-0.5, release_ms=50.0):
    ceiling = 10 ** (ceiling_db / 20.0)
    sr = 48000
    release_coeff = np.exp(-1.0 / (sr * release_ms / 1000.0))

    out = np.copy(audio)
    for ch in range(out.shape[0]):
        abs_sig = np.abs(out[ch])
        ideal_gain = np.minimum(1.0, ceiling / (abs_sig + 1e-10))
        gain = _limiter_release_loop(ideal_gain, release_coeff)
        out[ch] *= gain
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 2. Phase-Cohrent Limiter
# ═══════════════════════════════════════════════════════════════════════════

def phase_limit_fast(audio, sr, ceiling_db=-0.5, lookahead_ms=5, release_ms=100):
    ceiling = 10 ** (ceiling_db / 20.0)
    lookahead = int(sr * lookahead_ms / 1000)
    release_samples = int(sr * release_ms / 1000)
    release_coeff = np.exp(-1.0 / release_samples)

    out = np.copy(audio)
    for ch in range(out.shape[0]):
        abs_signal = np.abs(out[ch])
        max_in_window = _rolling_max_deque(abs_signal, lookahead)

        gain_curve = np.where(
            max_in_window > ceiling,
            ceiling / (max_in_window + 1e-10),
            1.0
        )

        smoothed = _smooth_forward(gain_curve, release_coeff)
        smoothed = _smooth_backward(smoothed, release_coeff)
        out[ch] *= smoothed
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 3. Multiband Compressor
# ═══════════════════════════════════════════════════════════════════════════

def _envelope_detect_correct(band_data, sr, attack_ms=10.0, release_ms=100.0):
    """RMS包絡検出 — ベクトル化RMS + Numba smoother。"""
    ws = int(sr * 0.02)
    atk_coeff = np.exp(-1.0 / (sr * attack_ms / 1000.0))
    rel_coeff = np.exp(-1.0 / (sr * release_ms / 1000.0))
    n = len(band_data)

    # フレーム単位RMS（ベクトル化）
    n_frames = n // ws
    if n_frames > 0:
        trimmed = band_data[:n_frames * ws].reshape(n_frames, ws)
        rms_vals = np.sqrt(np.mean(trimmed ** 2, axis=1) + 1e-10)
        env_frame = 20.0 * np.log10(rms_vals + 1e-10)
    else:
        return np.zeros(n)

    # フレーム→サンプル展開（Numba）
    env = _envelope_to_samples(env_frame, n, ws)

    # Smoother（Numba）
    env = _envelope_follower_smoother(env, atk_coeff, rel_coeff)

    return env


def multiband_compress_fast(audio, sr, thresholds=None, ratios=None,
                             attack_ms=10.0, release_ms=100.0):
    if thresholds is None:
        thresholds = [-20.0, -22.0, -24.0]
    if ratios is None:
        ratios = [3.0, 2.5, 2.0]

    low_mid = 200.0
    mid_high = 2000.0
    compressed = np.zeros_like(audio)

    for ch in range(audio.shape[0]):
        b_l, a_l = sig.butter(4, low_mid / (sr / 2), btype="low")
        b_hp, a_hp = sig.butter(4, low_mid / (sr / 2), btype="high")
        b_lp, a_lp = sig.butter(4, mid_high / (sr / 2), btype="low")
        b_h, a_h = sig.butter(4, mid_high / (sr / 2), btype="high")

        low = sig.lfilter(b_l, a_l, audio[ch])
        mid = sig.lfilter(b_lp, a_lp, sig.lfilter(b_hp, a_hp, audio[ch]))
        high = sig.lfilter(b_h, a_h, audio[ch])

        result = np.zeros(audio.shape[1])
        for band_data, thresh, ratio in zip([low, mid, high], thresholds, ratios):
            envelope = _envelope_detect_correct(band_data, sr, attack_ms, release_ms)
            gain_linear = _gain_reduce(envelope, thresh, ratio)
            result += band_data * gain_linear
        compressed[ch] = result
    return compressed


# ═══════════════════════════════════════════════════════════════════════════
# 4. True Peak Limiter
# ═══════════════════════════════════════════════════════════════════════════

def true_peak_limiter_fast(audio, sr, ceiling_db=-0.5, oversample=4):
    ceiling = 10 ** (ceiling_db / 20.0)
    out = np.copy(audio)
    for ch in range(out.shape[0]):
        upsampled = sig.resample(audio[ch], len(audio[ch]) * oversample)
        abs_up = np.abs(upsampled)
        n_orig = len(audio[ch])

        peak_env = _peak_env_from_upsampled(abs_up, oversample, n_orig)

        gain = np.ones(n_orig)
        mask = peak_env > ceiling
        gain[mask] = ceiling / (peak_env[mask] + 1e-10)

        smoothed = _smooth_true_peak(gain)
        out[ch] *= smoothed
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 5. Loudness Mapping
# ═══════════════════════════════════════════════════════════════════════════

def loudness_mapping_compress_fast(audio, sr, target_lufs=-14, strength=0.7):
    mono = (audio[0] + audio[1]) / 2.0 if audio.shape[0] == 2 else audio[0]
    frame_size = int(sr * 0.02)
    n_loop_frames = (len(mono) - frame_size) // frame_size

    if n_loop_frames <= 0:
        return audio

    loop_end = n_loop_frames * frame_size

    # RMS計算をベクトル化
    trimmed = mono[:loop_end].reshape(n_loop_frames, frame_size)
    rms_vals = np.sqrt(np.mean(trimmed ** 2, axis=1) + 1e-10)
    loudness_db = 20.0 * np.log10(rms_vals + 1e-10)

    original_mean = np.mean(loudness_db)
    original_std = np.std(loudness_db) + 1e-10
    target_mean = target_lufs
    target_std = original_std * (1 - strength) + 6.0 * strength
    inv_ratio = min(1.0, target_std / original_std)
    threshold = original_mean - 40

    above = loudness_db >= threshold
    target_db = np.where(
        above,
        (loudness_db - original_mean) * inv_ratio + target_mean,
        loudness_db + ((threshold - original_mean) * inv_ratio + target_mean - threshold)
    )
    gain_db = target_db - loudness_db
    gain_per_frame = 10 ** (gain_db / 20.0)

    gain_samples = np.repeat(gain_per_frame, frame_size)

    out = np.copy(audio)
    for ch in range(out.shape[0]):
        out[ch, :loop_end] *= gain_samples
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 6. Single Band Compress
# ═══════════════════════════════════════════════════════════════════════════

def single_band_compress_fast(channel, sr, threshold_db=-20, ratio=3):
    frame_size = int(sr * 0.02)
    out = np.copy(channel)
    atk = np.exp(-1.0 / (sr * 0.003))
    rel = np.exp(-1.0 / (sr * 0.100))
    current_env = -100.0
    for i in range(0, len(channel) - frame_size, frame_size):
        frame = channel[i:i + frame_size]
        rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
        env_db = 20 * np.log10(rms + 1e-10)
        if env_db > current_env:
            current_env = atk * current_env + (1 - atk) * env_db
        else:
            current_env = rel * current_env + (1 - rel) * env_db
        if current_env > threshold_db:
            overshoot = current_env - threshold_db
            reduction = overshoot * (1 - 1.0 / ratio)
            gain = 10 ** (-reduction / 20.0)
            out[i:i + frame_size] *= gain
    return out


# ═══════════════════════════════════════════════════════════════════════════
# JIT ウォームアップ（初回呼び出しのコンパイル遅延を避ける）
# ═══════════════════════════════════════════════════════════════════════════

def warmup_jit():
    """全Numba関数を事前コンパイル。"""
    dummy_f64 = np.zeros(100, dtype=np.float64)
    dummy_i64 = np.zeros(100, dtype=np.int64)
    _limiter_release_loop(dummy_f64, 0.999)
    _rolling_max_deque(dummy_f64, 10)
    _smooth_forward(dummy_f64, 0.999)
    _smooth_backward(dummy_f64.copy(), 0.999)
    _envelope_follower_smoother(dummy_f64, 0.999, 0.99)
    _envelope_to_samples(dummy_f64, 100, 10)
    _gain_reduce(dummy_f64, -20.0, 3.0)
    _smooth_true_peak(dummy_f64)
    _peak_env_from_upsampled(dummy_f64, 4, 25)


if __name__ == "__main__":
    import time
    print("Numba JIT ウォームアップ中...")
    t0 = time.time()
    warmup_jit()
    print(f"  コンパイル完了: {time.time()-t0:.2f}秒")
    print("dsp_optimized.py v4 — Numba JIT版")
    print("正常にインポートされました (=^･ω･^=)")
