#!/usr/bin/env python3
"""スタンドアロンベンチマーク — 元の関数 vs 最適化版"""
import time, sys, os, numpy as np
from scipy import signal

sys.path.insert(0, '/opt/data/remaster')

# ═══════════════════════════════════════════════════════════════════
# 元の関数（app.pyから直接コピー — importエラー回避）
# ═══════════════════════════════════════════════════════════════════

def design_biquad(filter_type, freq, sr, gain_db=0.0, q=0.707):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    if filter_type == "low_shelf":
        b0 = A*((A+1)-(A-1)*np.cos(w0)+2*np.sqrt(A)*alpha)
        b1 = 2*A*((A-1)-(A+1)*np.cos(w0))
        b2 = A*((A+1)-(A-1)*np.cos(w0)-2*np.sqrt(A)*alpha)
        a0 = (A+1)+(A-1)*np.cos(w0)+2*np.sqrt(A)*alpha
        a1 = -2*((A-1)+(A+1)*np.cos(w0))
        a2 = (A+1)+(A-1)*np.cos(w0)-2*np.sqrt(A)*alpha
    elif filter_type == "high_shelf":
        b0 = A*((A+1)+(A-1)*np.cos(w0)+2*np.sqrt(A)*alpha)
        b1 = -2*A*((A-1)+(A+1)*np.cos(w0))
        b2 = A*((A+1)+(A-1)*np.cos(w0)-2*np.sqrt(A)*alpha)
        a0 = (A+1)-(A-1)*np.cos(w0)+2*np.sqrt(A)*alpha
        a1 = 2*((A-1)-(A+1)*np.cos(w0))
        a2 = (A+1)-(A-1)*np.cos(w0)-2*np.sqrt(A)*alpha
    elif filter_type == "peaking":
        b0 = 1+alpha*A; b1 = -2*np.cos(w0); b2 = 1-alpha*A
        a0 = 1+alpha/A; a1 = -2*np.cos(w0); a2 = 1-alpha/A
    else:
        raise ValueError(f"Unknown filter: {filter_type}")
    return np.array([b0,b1,b2])/a0, np.array([1.0, a1/a0, a2/a0])

def apply_iir(audio, b, a):
    out = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        out[ch] = signal.lfilter(b, a, audio[ch])
    return out

# --- 元の limiter ---
def limiter_orig(audio, ceiling_db=-0.5, release_ms=50.0):
    ceiling = 10**(ceiling_db/20.0)
    rel = np.exp(-1.0/(48000*release_ms/1000.0))
    out = np.copy(audio)
    for ch in range(out.shape[0]):
        g = 1.0
        for i in range(out.shape[1]):
            av = abs(out[ch,i])
            if av*g > ceiling:
                g = min(g, ceiling/(av+1e-10))
            else:
                g = min(1.0, g*rel+(1-rel))
            out[ch,i] *= g
    return out

# --- 元の phase_limit ---
def phase_limit_orig(audio, sr, ceiling_db=-0.5, lookahead_ms=5, release_ms=100):
    ceiling = 10**(ceiling_db/20.0)
    lookahead = int(sr*lookahead_ms/1000)
    release_samples = int(sr*release_ms/1000)
    release_coeff = np.exp(-1.0/release_samples)
    out = np.copy(audio)
    for ch in range(out.shape[0]):
        abs_signal = np.abs(out[ch])
        gain_curve = np.ones(len(abs_signal))
        for i in range(len(abs_signal)):
            window_end = min(i+lookahead, len(abs_signal))
            max_in_window = np.max(abs_signal[i:window_end])
            if max_in_window > ceiling:
                gain_curve[i] = ceiling/(max_in_window+1e-10)
        smoothed = np.copy(gain_curve)
        for i in range(1, len(smoothed)):
            if smoothed[i] > smoothed[i-1]:
                smoothed[i] = smoothed[i-1]*release_coeff+smoothed[i]*(1-release_coeff)
        for i in range(len(smoothed)-2, -1, -1):
            if smoothed[i] > smoothed[i+1]:
                smoothed[i] = smoothed[i+1]*release_coeff+smoothed[i]*(1-release_coeff)
        out[ch] *= smoothed
    return out

# --- 元の multiband_compress ---
def multiband_compress_orig(audio, sr, thresholds=None, ratios=None):
    if thresholds is None: thresholds = [-20.0, -22.0, -24.0]
    if ratios is None: ratios = [3.0, 2.5, 2.0]
    low_mid, mid_high = 200.0, 2000.0
    compressed = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        b_l,a_l = signal.butter(4, low_mid/(sr/2), btype="low")
        low = signal.lfilter(b_l, a_l, audio[ch])
        b_hp,a_hp = signal.butter(4, low_mid/(sr/2), btype="high")
        b_lp,a_lp = signal.butter(4, mid_high/(sr/2), btype="low")
        mid = signal.lfilter(b_hp, a_hp, audio[ch])
        mid = signal.lfilter(b_lp, a_lp, mid)
        b_h,a_h = signal.butter(4, mid_high/(sr/2), btype="high")
        high = signal.lfilter(b_h, a_h, audio[ch])
        atk = np.exp(-1.0/(sr*0.010))
        rel = np.exp(-1.0/(sr*0.100))
        result = np.zeros(audio.shape[1])
        for band_data, thresh, ratio in zip([low,mid,high], thresholds, ratios):
            ws = int(sr*0.02)
            env = np.zeros(len(band_data))
            for i in range(0, len(band_data), ws):
                chunk = band_data[i:i+ws]
                rms = np.sqrt(np.mean(chunk**2)+1e-10)
                env[i:i+ws] = 20*np.log10(rms+1e-10)
            cur = -100.0
            gain = np.zeros(len(band_data))
            for i in range(len(band_data)):
                if env[i] > cur: cur = atk*cur+(1-atk)*env[i]
                else: cur = rel*cur+(1-rel)*env[i]
                if cur > thresh: gain[i] = -(cur-thresh)*(1-1.0/ratio)
            result += band_data * 10**(gain/20.0)
        compressed[ch] = result
    return compressed

# --- 元の loudness_mapping_compress ---
def loudness_mapping_compress_orig(audio, sr, target_lufs=-14, strength=0.7):
    mono = (audio[0]+audio[1])/2.0 if audio.shape[0]==2 else audio[0]
    frame_size = int(sr*0.02)
    loudness_values = []
    for i in range(0, len(mono)-frame_size, frame_size):
        frame = mono[i:i+frame_size]
        rms = np.sqrt(np.mean(frame**2)+1e-10)
        loudness_values.append(20*np.log10(rms+1e-10))
    if not loudness_values: return audio
    loudness_arr = np.array(loudness_values)
    original_mean = np.mean(loudness_arr)
    original_std = np.std(loudness_arr)+1e-10
    target_mean = target_lufs
    target_std = original_std*(1-strength)+6.0*strength
    inv_ratio = min(1.0, target_std/original_std)
    threshold = original_mean-40
    def map_loudness(x):
        if x >= threshold:
            return (x-original_mean)*inv_ratio+target_mean
        else:
            return x+((threshold-original_mean)*inv_ratio+target_mean-threshold)
    out = np.copy(audio)
    for ch in range(out.shape[0]):
        for i in range(0, len(mono)-frame_size, frame_size):
            frame = mono[i:i+frame_size]
            rms = np.sqrt(np.mean(frame**2)+1e-10)
            current_db = 20*np.log10(rms+1e-10)
            target_db = map_loudness(current_db)
            gain_db = target_db-current_db
            gain = 10**(gain_db/20.0)
            out[ch,i:i+frame_size] *= gain
    return out

# --- 元の single_band_compress ---
def single_band_compress_orig(channel, sr, threshold_db=-20, ratio=3):
    frame_size = int(sr*0.02)
    out = np.copy(channel)
    atk = np.exp(-1.0/(sr*0.003))
    rel = np.exp(-1.0/(sr*0.100))
    current_env = -100.0
    for i in range(0, len(channel)-frame_size, frame_size):
        frame = channel[i:i+frame_size]
        rms = np.sqrt(np.mean(frame**2)+1e-10)
        env_db = 20*np.log10(rms+1e-10)
        if env_db > current_env: current_env = atk*current_env+(1-atk)*env_db
        else: current_env = rel*current_env+(1-rel)*env_db
        if current_env > threshold_db:
            overshoot = current_env-threshold_db
            reduction = overshoot*(1-1.0/ratio)
            gain = 10**(-reduction/20.0)
            out[i:i+frame_size] *= gain
    return out

# ═══════════════════════════════════════════════════════════════════
# テスト音声生成
# ═══════════════════════════════════════════════════════════════════
sr = 48000; duration = 30; n = sr * duration
np.random.seed(42)
audio = np.random.randn(2, n) * 0.1
t = np.linspace(0, duration, n)
audio[0] += 0.3 * np.sin(2*np.pi*440*t)
audio[1] += 0.3 * np.sin(2*np.pi*442*t)
audio = np.clip(audio, -0.95, 0.95)

print(f"テスト音声: {duration}s, {sr}Hz, stereo ({n:,} samples/ch)")
print("=" * 65)

results = []

# Limiter
print("\n[1/5] Limiter...")
t0=time.time(); r1=limiter_orig(audio.copy(),-0.5,50.0); t1=time.time()-t0
from dsp_optimized import limiter_fast
t0=time.time(); r2=limiter_fast(audio.copy(),-0.5,50.0); t2=time.time()-t0
speedup = t1/t2 if t2>0 else 0
diff = np.max(np.abs(r1-r2))
print(f"  元:     {t1:.2f}s")
print(f"  最適化: {t2:.2f}s  ({speedup:.1f}x faster)")
print(f"  最大差: {diff:.2e}")
results.append(('Limiter', t1, t2, speedup, diff))

# Phase Limit
print("\n[2/5] Phase Limit...")
t0=time.time(); p1=phase_limit_orig(audio.copy(),sr,-0.5,5,100); t1=time.time()-t0
from dsp_optimized import phase_limit_fast
t0=time.time(); p2=phase_limit_fast(audio.copy(),sr,-0.5,5,100); t2=time.time()-t0
speedup = t1/t2 if t2>0 else 0
diff = np.max(np.abs(p1-p2))
print(f"  元:     {t1:.2f}s")
print(f"  最適化: {t2:.2f}s  ({speedup:.1f}x faster)")
print(f"  最大差: {diff:.2e}")
results.append(('Phase Limit', t1, t2, speedup, diff))

# Multiband Compress
print("\n[3/5] Multiband Compress...")
t0=time.time(); m1=multiband_compress_orig(audio.copy(),sr); t1=time.time()-t0
from dsp_optimized import multiband_compress_fast
t0=time.time(); m2=multiband_compress_fast(audio.copy(),sr); t2=time.time()-t0
speedup = t1/t2 if t2>0 else 0
diff = np.max(np.abs(m1-m2))
print(f"  元:     {t1:.2f}s")
print(f"  最適化: {t2:.2f}s  ({speedup:.1f}x faster)")
print(f"  最大差: {diff:.2e}")
results.append(('Multiband Comp', t1, t2, speedup, diff))

# Loudness Mapping
print("\n[4/5] Loudness Mapping...")
t0=time.time(); l1=loudness_mapping_compress_orig(audio.copy(),sr,-14,0.7); t1=time.time()-t0
from dsp_optimized import loudness_mapping_compress_fast
t0=time.time(); l2=loudness_mapping_compress_fast(audio.copy(),sr,-14,0.7); t2=time.time()-t0
speedup = t1/t2 if t2>0 else 0
diff = np.max(np.abs(l1-l2))
print(f"  元:     {t1:.2f}s")
print(f"  最適化: {t2:.2f}s  ({speedup:.1f}x faster)")
print(f"  最大差: {diff:.2e}")
results.append(('Loudness Map', t1, t2, speedup, diff))

# Single Band
print("\n[5/5] Single Band Compress...")
ch = audio[0].copy()
t0=time.time(); s1=single_band_compress_orig(ch.copy(),sr,-20,3); t1=time.time()-t0
from dsp_optimized import single_band_compress_fast
t0=time.time(); s2=single_band_compress_fast(ch.copy(),sr,-20,3); t2=time.time()-t0
speedup = t1/t2 if t2>0 else 0
diff = np.max(np.abs(s1-s2))
print(f"  元:     {t1:.2f}s")
print(f"  最適化: {t2:.2f}s  ({speedup:.1f}x faster)")
print(f"  最大差: {diff:.2e}")
results.append(('Single Band', t1, t2, speedup, diff))

# サマリー
print("\n" + "=" * 65)
print("サマリー")
print("=" * 65)
total_orig = sum(r[1] for r in results)
total_fast = sum(r[2] for r in results)
for name, t_orig, t_fast, spd, diff in results:
    print(f"  {name:18s}: {t_orig:7.2f}s → {t_fast:7.2f}s  ({spd:5.1f}x)  diff={diff:.2e}")
print(f"\n  {'合計':18s}: {total_orig:7.2f}s → {total_fast:7.2f}s  ({total_orig/total_fast:.1f}x)")
print("\n完了！ (=^･ω･^=)")
