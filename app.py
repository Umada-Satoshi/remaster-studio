#!/usr/bin/env python3
"""
Remaster Studio — 高音質 重低音重視 リマスターツール (Web UI)
=============================================================
Full-featured audio remastering with parametric EQ,
multiband compression, limiter, LUFS normalization, and spectrum analysis.

Usage:
    python3 app.py
    # → http://localhost:7860
"""

import io
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
import wave
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file, render_template_string
from scipy import signal

app = Flask(__name__)
# Fixed paths — all gunicorn workers share the same filesystem
_DATA = Path(os.environ.get("REMASTER_DATA_DIR", "/app/data"))
UPLOAD_DIR = _DATA / "uploads"
OUTPUT_DIR = _DATA / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB

# ── AI Analysis Engine ─────────────────────────────────────────────────
# Analyzes audio characteristics and recommends optimal mastering settings.
# Based on professional mastering engineer decision trees.

def classify_genre(audio, sr, analysis):
    """Classify audio genre from spectral/dynamic characteristics."""
    bands = analysis["bands"]
    crest = analysis["crest_db"]
    rms = analysis["rms_db"]

    sub = bands["sub_bass_20_60"]
    bass = bands["bass_60_200"]
    low_mid = bands["low_mid_200_500"]
    mid = bands["mid_500_2k"]
    upper = bands["upper_mid_2k_6k"]
    high = bands["high_6k_12k"]
    air = bands["air_12k_20k"]

    # Relative band strengths (normalized)
    total = max(sub, bass, low_mid, mid, upper, high, air) + 60  # shift to positive
    if total <= 0:
        return "default", 0.5

    sub_r = max(0, sub + 60) / total
    bass_r = max(0, bass + 60) / total
    mid_r = max(0, mid + 60) / total
    upper_r = max(0, upper + 60) / total
    high_r = max(0, high + 60) / total
    air_r = max(0, air + 60) / total

    scores = {}

    # Hip-Hop: heavy sub-bass, scooped mids, moderate highs
    scores["hiphop"] = (sub_r * 3 + bass_r * 1.5 - mid_r * 0.5 + (1 if crest < 8 else 0)) * 10

    # EDM: extreme sub, punchy upper-mid, bright air
    scores["edm"] = (sub_r * 3.5 + air_r * 1.5 + upper_r - mid_r * 0.3) * 10

    # Pop: balanced with vocal presence (3-5kHz)
    scores["pop"] = (upper_r * 2.5 + mid_r * 1.5 + high_r - sub_r * 0.5) * 10

    # Rock: strong mids and low-mids, moderate bass
    scores["rock"] = (mid_r * 2 + low_mid/60 * 1.5 + upper_r * 1.5 - sub_r * 0.3) * 10

    # Jazz: balanced, natural dynamics (high crest factor)
    scores["jazz"] = (crest * 0.5 + (1 if crest > 12 else 0) * 5 + mid_r * 1.5) * 10

    # Classical: very wide dynamics, balanced spectrum
    scores["classical"] = (crest * 0.8 + (1 if crest > 15 else 0) * 8) * 10

    # R&B: warm low-mids, smooth highs
    scores["rnb"] = (bass_r * 2 + low_mid/60 * 1.5 + mid_r - high_r * 0.3) * 10

    # Lo-Fi: rolled-off highs, boosted low-mids
    scores["lofi"] = ((1 if high < -50 else 0) * 5 + low_mid/60 * 2 + bass_r * 1.5) * 10

    # Bass Music: extreme sub-bass dominance
    scores["bassmusic"] = (sub_r * 5 + bass_r * 2 - mid_r) * 10

    # Podcast: vocal-centric, minimal bass
    scores["podcast"] = (upper_r * 3 + mid_r * 2 - sub_r * 2) * 10

    # Cinematic: wide dynamics, deep sub
    scores["cinematic"] = (crest * 0.6 + sub_r * 2 + air_r * 1.5) * 10

    best = max(scores, key=scores.get)
    confidence = min(1.0, scores[best] / 50)
    return best, round(confidence, 2)


def recommend_settings(analysis, genre, confidence):
    """Recommend optimal mastering parameters based on analysis + genre."""
    bands = analysis["bands"]
    crest = analysis["crest_db"]
    rms = analysis["rms_db"]
    peak = analysis["peak_db"]

    sub = bands["sub_bass_20_60"]
    bass = bands["bass_60_200"]
    mid = bands["mid_500_2k"]
    high = bands["high_6k_12k"]

    # Start from genre preset
    from copy import deepcopy
    base_presets = {
        "hiphop":    {"subBass":10,"bass":8,"bassFreq":120,"subFreq":45,"mid":-2,"midFreq":300,"high":3,"highFreq":4000,"compTh":-16,"compRa":5,"lim":-0.3,"lufs":-9,"stereo":1.15,"sampleRate":48000},
        "edm":       {"subBass":12,"bass":10,"bassFreq":100,"subFreq":40,"mid":1,"midFreq":2000,"high":2,"highFreq":10000,"compTh":-14,"compRa":6,"lim":-0.2,"lufs":-8,"stereo":1.4,"sampleRate":48000},
        "pop":       {"subBass":3,"bass":4,"bassFreq":120,"subFreq":60,"mid":3,"midFreq":3500,"high":2.5,"highFreq":10000,"compTh":-18,"compRa":3,"lim":-0.5,"lufs":-12,"stereo":1.2,"sampleRate":48000},
        "rock":      {"subBass":5,"bass":7,"bassFreq":100,"subFreq":50,"mid":3,"midFreq":2500,"high":2,"highFreq":6000,"compTh":-16,"compRa":4,"lim":-0.5,"lufs":-10,"stereo":1.3,"sampleRate":48000},
        "jazz":      {"subBass":2,"bass":3,"bassFreq":120,"subFreq":60,"mid":1,"midFreq":2000,"high":1,"highFreq":6000,"compTh":-24,"compRa":2,"lim":-1,"lufs":-16,"stereo":1.15,"sampleRate":96000},
        "classical": {"subBass":1,"bass":1,"bassFreq":80,"subFreq":40,"mid":0,"midFreq":1000,"high":0.5,"highFreq":12000,"compTh":-30,"compRa":1.5,"lim":-2,"lufs":-24,"stereo":1.5,"sampleRate":96000},
        "rnb":       {"subBass":4,"bass":5,"bassFreq":130,"subFreq":55,"mid":1,"midFreq":2500,"high":1.5,"highFreq":7000,"compTh":-18,"compRa":3,"lim":-0.5,"lufs":-12,"stereo":1.15,"sampleRate":48000},
        "lofi":      {"subBass":5,"bass":6,"bassFreq":150,"subFreq":70,"mid":2,"midFreq":500,"high":-1,"highFreq":4000,"compTh":-16,"compRa":3,"lim":-1,"lufs":-14,"stereo":1.0,"sampleRate":44100},
        "bassmusic": {"subBass":15,"bass":12,"bassFreq":80,"subFreq":35,"mid":-1,"midFreq":2000,"high":2,"highFreq":8000,"compTh":-12,"compRa":7,"lim":-0.1,"lufs":-7,"stereo":1.3,"sampleRate":48000},
        "podcast":   {"subBass":-2,"bass":1,"bassFreq":100,"subFreq":80,"mid":4,"midFreq":3000,"high":2,"highFreq":5000,"compTh":-20,"compRa":4,"lim":-0.5,"lufs":-16,"stereo":1.0,"sampleRate":48000},
        "cinematic": {"subBass":8,"bass":6,"bassFreq":80,"subFreq":30,"mid":2,"midFreq":1500,"high":2,"highFreq":12000,"compTh":-22,"compRa":2.5,"lim":-1,"lufs":-20,"stereo":1.5,"sampleRate":48000},
        "default":   {"subBass":4,"bass":6,"bassFreq":150,"subFreq":60,"mid":2,"midFreq":3000,"high":1.5,"highFreq":8000,"compTh":-20,"compRa":3,"lim":-0.5,"lufs":-14,"stereo":1.2,"sampleRate":48000},
    }
    p = deepcopy(base_presets.get(genre, base_presets["default"]))

    # Adaptive adjustments based on actual audio analysis
    # 1. Sub-bass compensation: if sub is weak, boost more; if strong, ease off
    if sub < -50:
        p["subBass"] = min(18, p["subBass"] + 3)
    elif sub > -30:
        p["subBass"] = max(0, p["subBass"] - 2)

    # 2. Bass compensation
    if bass < -50:
        p["bass"] = min(18, p["bass"] + 3)
    elif bass > -25:
        p["bass"] = max(0, p["bass"] - 2)

    # 3. Dynamic range adaptation: if too compressed, ease compression
    if crest < 4:
        p["compTh"] = min(-10, p["compTh"] + 4)
        p["compRa"] = max(1.5, p["compRa"] - 1)
    elif crest > 18:
        p["compTh"] = max(-30, p["compTh"] - 3)
        p["compRa"] = min(7, p["compRa"] + 0.5)

    # 4. Loudness normalization target
    # If already loud (high RMS), don't push too hard
    if rms > -8:
        p["lufs"] = max(-14, p["lufs"] + 2)
        p["lim"] = max(-1.0, p["lim"] - 0.3)

    # 5. High frequency compensation
    if high < -60:
        p["high"] = min(12, p["high"] + 2)

    # 6. Stereo width: wider for acoustic, narrower for bass-heavy
    if genre in ("classical", "jazz", "cinematic"):
        p["stereo"] = min(2.0, p["stereo"] + 0.1)
    elif genre in ("bassmusic", "hiphop", "lofi"):
        p["stereo"] = max(0.8, p["stereo"] - 0.1)

    # Round all values
    for k in p:
        if isinstance(p[k], float):
            p[k] = round(p[k], 1)

    # Build reasoning
    reasoning = []
    reasoning.append(f"ジャンル判定: {genre} (信頼度 {confidence*100:.0f}%)")
    if sub < -50:
        reasoning.append("サブベースが弱い → +3dBブースト適用")
    if crest < 4:
        reasoning.append("ダイナミクスが潰れている → コンプ緩和")
    elif crest > 18:
        reasoning.append("ダイナミクスが豊か → コンプ強化")
    if rms > -8:
        reasoning.append("既にラウド → LUFS目標を緩和")
    if high < -60:
        reasoning.append("高域が暗い → ハイブースト追加")

    p["_reasoning"] = reasoning
    p["_genre"] = genre
    p["_confidence"] = confidence
    return p


# ── DSP Core ─────────────────────────────────────────────────────────────

def design_biquad(filter_type, freq, sr, gain_db=0.0, q=0.707):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    if filter_type == "low_shelf":
        b0 = A * ((A+1)-(A-1)*np.cos(w0)+2*np.sqrt(A)*alpha)
        b1 = 2*A*((A-1)-(A+1)*np.cos(w0))
        b2 = A * ((A+1)-(A-1)*np.cos(w0)-2*np.sqrt(A)*alpha)
        a0 = (A+1)+(A-1)*np.cos(w0)+2*np.sqrt(A)*alpha
        a1 = -2*((A-1)+(A+1)*np.cos(w0))
        a2 = (A+1)+(A-1)*np.cos(w0)-2*np.sqrt(A)*alpha
    elif filter_type == "high_shelf":
        b0 = A * ((A+1)+(A-1)*np.cos(w0)+2*np.sqrt(A)*alpha)
        b1 = -2*A*((A-1)+(A+1)*np.cos(w0))
        b2 = A * ((A+1)+(A-1)*np.cos(w0)-2*np.sqrt(A)*alpha)
        a0 = (A+1)-(A-1)*np.cos(w0)+2*np.sqrt(A)*alpha
        a1 = 2*((A-1)-(A+1)*np.cos(w0))
        a2 = (A+1)-(A-1)*np.cos(w0)-2*np.sqrt(A)*alpha
    elif filter_type == "peaking":
        b0 = 1+alpha*A
        b1 = -2*np.cos(w0)
        b2 = 1-alpha*A
        a0 = 1+alpha/A
        a1 = -2*np.cos(w0)
        a2 = 1-alpha/A
    else:
        raise ValueError(f"Unknown filter: {filter_type}")
    return np.array([b0,b1,b2])/a0, np.array([1.0, a1/a0, a2/a0])


def apply_iir(audio, b, a):
    out = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        out[ch] = signal.lfilter(b, a, audio[ch])
    return out


def multiband_compress(audio, sr, thresholds=None, ratios=None):
    if thresholds is None:
        thresholds = [-20.0, -22.0, -24.0]
    if ratios is None:
        ratios = [3.0, 2.5, 2.0]
    low_mid, mid_high = 200.0, 2000.0
    bands = []
    for ch in range(audio.shape[0]):
        b_l, a_l = signal.butter(4, low_mid/(sr/2), btype="low")
        low = signal.lfilter(b_l, a_l, audio[ch])
        b_hp, a_hp = signal.butter(4, low_mid/(sr/2), btype="high")
        b_lp, a_lp = signal.butter(4, mid_high/(sr/2), btype="low")
        mid = signal.lfilter(b_hp, a_hp, audio[ch])
        mid = signal.lfilter(b_lp, a_lp, mid)
        b_h, a_h = signal.butter(4, mid_high/(sr/2), btype="high")
        high = signal.lfilter(b_h, a_h, audio[ch])
        bands.append((low, mid, high))

    atk = np.exp(-1.0/(sr*0.010))
    rel = np.exp(-1.0/(sr*0.100))
    compressed = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        result = np.zeros(audio.shape[1])
        for band_data, thresh, ratio in zip(bands[ch], thresholds, ratios):
            ws = int(sr*0.02)
            env = np.zeros(len(band_data))
            for i in range(0, len(band_data), ws):
                chunk = band_data[i:i+ws]
                rms = np.sqrt(np.mean(chunk**2)+1e-10)
                env[i:i+ws] = 20*np.log10(rms+1e-10)
            cur = -100.0
            gain = np.zeros(len(band_data))
            for i in range(len(band_data)):
                if env[i] > cur:
                    cur = atk*cur+(1-atk)*env[i]
                else:
                    cur = rel*cur+(1-rel)*env[i]
                if cur > thresh:
                    gain[i] = -(cur-thresh)*(1-1.0/ratio)
            result += band_data * 10**(gain/20.0)
        compressed[ch] = result
    return compressed


def limiter(audio, ceiling_db=-0.5, release_ms=50.0):
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


def normalize_lufs(audio, sr, target_lufs):
    b_hp, a_hp = signal.butter(2, 38.0/(sr/2), btype="high")
    b_sh, a_sh = design_biquad("high_shelf", 2500.0, sr, 4.0, 0.707)
    weighted = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        w = signal.lfilter(b_hp, a_hp, audio[ch])
        w = signal.lfilter(b_sh, a_sh, w)
        weighted[ch] = w
    gate = 10**((-70.0)/10.0)
    ss, cnt = 0.0, 0
    for ch in range(weighted.shape[0]):
        for i in range(0, weighted.shape[1], sr):
            blk = weighted[ch,i:i+sr]
            if len(blk)>0:
                ms = np.mean(blk**2)
                if ms > gate:
                    ss += ms; cnt += 1
    if cnt == 0: return audio
    lufs = -0.691+10*np.log10(ss/cnt+1e-10)
    return audio * 10**((target_lufs-lufs)/20.0)


def stereo_enhance(audio, width=1.2):
    if audio.shape[0] != 2: return audio
    mid = (audio[0]+audio[1])/2.0
    side = (audio[0]-audio[1])/2.0 * width
    return np.array([mid+side, mid-side])


def analyze_spectrum(audio, sr):
    mono = (audio[0]+audio[1])/2.0 if audio.shape[0]==2 else audio[0]
    nperseg = min(len(mono), sr*4)
    f, t, Zxx = signal.stft(mono, fs=sr, nperseg=nperseg)
    power = np.mean(np.abs(Zxx)**2, axis=1)
    def be(lo,hi):
        m = (f>=lo)&(f<hi)
        return float(10*np.log10(np.mean(power[m])+1e-10)) if m.any() else -100.0
    rms = float(np.sqrt(np.mean(mono**2)))
    peak = float(np.max(np.abs(mono)))
    # Full spectrum for plotting
    freqs = f.tolist()
    power_list = (10*np.log10(power+1e-10)).tolist()
    return {
        "sample_rate": sr,
        "duration": round(len(mono)/sr, 2),
        "channels": "stereo" if audio.shape[0]==2 else "mono",
        "rms_db": round(20*np.log10(rms+1e-10), 2),
        "peak_db": round(20*np.log10(peak+1e-10), 2),
        "crest_db": round(20*np.log10(peak+1e-10)-20*np.log10(rms+1e-10), 2),
        "bands": {
            "sub_bass_20_60": round(be(20,60),1),
            "bass_60_200": round(be(60,200),1),
            "low_mid_200_500": round(be(200,500),1),
            "mid_500_2k": round(be(500,2000),1),
            "upper_mid_2k_6k": round(be(2000,6000),1),
            "high_6k_12k": round(be(6000,12000),1),
            "air_12k_20k": round(be(12000,20000),1),
        },
        "spectrum": {
            "freqs": freqs[::10],
            "power": [round(p,1) for p in power_list[::10]],
        }
    }


def read_audio(path, sr=48000):
    tmp = str(UPLOAD_DIR / f"conv_{uuid.uuid4().hex}.wav")
    subprocess.run(["ffmpeg","-y","-i",str(path),"-ar",str(sr),"-ac","2","-sample_fmt","s16",tmp],
                   capture_output=True, check=True)
    with wave.open(tmp,"rb") as wf:
        n = wf.getnframes()
        raw = wf.readframes(n)
        s = wf.getframerate()
    os.unlink(tmp)
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float64).reshape(-1,2).T / 32768.0
    return a, s


def write_audio(audio, sr, path, fmt="wav", bitrate="320k"):
    audio = np.clip(audio, -1.0, 1.0)
    ai = (audio*32767).astype(np.int16).T
    tmp = str(OUTPUT_DIR / f"enc_{uuid.uuid4().hex}.wav")
    with wave.open(tmp,"wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(ai.tobytes())
    cmd = ["ffmpeg","-y","-i",tmp]
    if fmt=="mp3": cmd += ["-codec:a","libmp3lame","-b:a",bitrate]
    elif fmt=="flac": cmd += ["-codec:a","flac"]
    elif fmt=="ogg": cmd += ["-codec:a","libvorbis","-q:a","10"]
    else: cmd += ["-codec:a","pcm_s16le"]
    cmd.append(str(path))
    subprocess.run(cmd, capture_output=True, check=True)
    os.unlink(tmp)


# ── PhaseLimiter-derived DSP ─────────────────────────────────────────
# Algorithms inspired by ai-mastering/phaselimiter (MIT License)
# https://github.com/ai-mastering/phaselimiter

def phase_limit(audio, sr, ceiling_db=-0.5, lookahead_ms=5, release_ms=100):
    """Phase-coherent limiter — minimizes gain reduction artifacts.
    Unlike traditional limiters, this preserves phase relationships
    by using smooth gain curves derived from the envelope."""
    ceiling = 10 ** (ceiling_db / 20.0)
    lookahead = int(sr * lookahead_ms / 1000)
    release_samples = int(sr * release_ms / 1000)
    release_coeff = np.exp(-1.0 / release_samples)

    out = np.copy(audio)
    for ch in range(out.shape[0]):
        # Forward peak detection with lookahead
        abs_signal = np.abs(out[ch])
        # Find peaks in lookahead window
        gain_curve = np.ones(len(abs_signal))
        for i in range(len(abs_signal)):
            window_end = min(i + lookahead, len(abs_signal))
            max_in_window = np.max(abs_signal[i:window_end])
            if max_in_window > ceiling:
                gain_curve[i] = ceiling / (max_in_window + 1e-10)

        # Smooth the gain curve (phase-coherent approach)
        smoothed = np.copy(gain_curve)
        # Forward smoothing
        for i in range(1, len(smoothed)):
            if smoothed[i] > smoothed[i-1]:
                smoothed[i] = smoothed[i-1] * release_coeff + smoothed[i] * (1 - release_coeff)
        # Backward smoothing (lookahead)
        for i in range(len(smoothed) - 2, -1, -1):
            if smoothed[i] > smoothed[i+1]:
                smoothed[i] = smoothed[i+1] * release_coeff + smoothed[i] * (1 - release_coeff)

        out[ch] *= smoothed
    return out


def loudness_mapping_compress(audio, sr, target_lufs=-14, strength=0.7):
    """Statistical loudness distribution mapping (phaselimiter LoudnessMapping).
    Maps input loudness distribution to target distribution.
    More natural than traditional threshold/ratio compression."""
    mono = (audio[0] + audio[1]) / 2.0 if audio.shape[0] == 2 else audio[0]

    # Calculate loudness histogram (frame-based)
    frame_size = int(sr * 0.02)  # 20ms frames
    loudness_values = []
    for i in range(0, len(mono) - frame_size, frame_size):
        frame = mono[i:i + frame_size]
        rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
        loudness_db = 20 * np.log10(rms + 1e-10)
        loudness_values.append(loudness_db)

    if not loudness_values:
        return audio

    loudness_arr = np.array(loudness_values)
    original_mean = np.mean(loudness_arr)
    original_std = np.std(loudness_arr) + 1e-10

    # Target distribution
    target_mean = target_lufs
    target_std = original_std * (1 - strength) + 6.0 * strength  # target ~6dB dynamic range

    # Loudness mapping function
    inv_ratio = min(1.0, target_std / original_std)
    threshold = original_mean - 40

    def map_loudness(x):
        if x >= threshold:
            return (x - original_mean) * inv_ratio + target_mean
        else:
            gain = (threshold - original_mean) * inv_ratio + target_mean - threshold
            return x + gain

    # Apply frame-wise gain
    out = np.copy(audio)
    for ch in range(out.shape[0]):
        for i in range(0, len(mono) - frame_size, frame_size):
            frame = mono[i:i + frame_size]
            rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
            current_db = 20 * np.log10(rms + 1e-10)
            target_db = map_loudness(current_db)
            gain_db = target_db - current_db
            gain = 10 ** (gain_db / 20.0)
            out[ch, i:i + frame_size] *= gain

    return out


def ms_compressor(audio, sr, threshold_db=-20, ratio=3, side_boost_db=0):
    """Mid/Side domain compression (phaselimiter MsCompressor).
    Processes mid and side channels independently for better
    stereo image control."""
    if audio.shape[0] != 2:
        return audio

    # Convert to M/S
    mid = (audio[0] + audio[1]) / 2.0
    side = (audio[0] - audio[1]) / 2.0

    # Compress mid channel
    mid_compressed = single_band_compress(mid, sr, threshold_db, ratio)

    # Process side channel (boost or compress)
    if side_boost_db != 0:
        b, a = design_biquad("peaking", 2000, sr, side_boost_db, 0.707)
        side = apply_iir(side.reshape(1, -1), b, a)[0]
    side_compressed = single_band_compress(side, sr, threshold_db - 3, ratio * 0.5)

    # Convert back to L/R
    left = mid_compressed + side_compressed
    right = mid_compressed - side_compressed
    return np.array([left, right])


def single_band_compress(channel, sr, threshold_db=-20, ratio=3):
    """Single-band compressor for one channel."""
    frame_size = int(sr * 0.02)
    out = np.copy(channel)
    atk = np.exp(-1.0 / (sr * 0.003))  # 3ms attack
    rel = np.exp(-1.0 / (sr * 0.100))  # 100ms release

    current_env = -100.0
    for i in range(0, len(channel) - frame_size, frame_size):
        frame = channel[i:i + frame_size]
        rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
        env_db = 20 * np.log10(rms + 1e-10)

        # Envelope follower
        if env_db > current_env:
            current_env = atk * current_env + (1 - atk) * env_db
        else:
            current_env = rel * current_env + (1 - rel) * env_db

        # Gain reduction
        if current_env > threshold_db:
            overshoot = current_env - threshold_db
            reduction = overshoot * (1 - 1.0 / ratio)
            gain = 10 ** (-reduction / 20.0)
            out[i:i + frame_size] *= gain

    return out


def highpass_fir(audio, sr, cutoff_freq=20, attenuation_db=70):
    """FIR high-pass filter for rumble removal (phaselimiter CutLowAndHighFreq).
    Uses Keiser window design for clean stopband attenuation."""
    if cutoff_freq <= 0:
        return audio

    normalized_freq = cutoff_freq / sr
    transition_width = 5.0 / sr

    # Keiser window parameters
    alpha = 0.1102 * (attenuation_db - 8.7)
    filter_len = int((attenuation_db - 7.95) / (2.285 * np.pi * transition_width)) + 1
    filter_len = max(filter_len, 64)
    if filter_len % 2 == 0:
        filter_len += 1

    # Design FIR bandpass (high-pass = bandpass from cutoff to Nyquist)
    n = np.arange(filter_len)
    mid = (filter_len - 1) / 2.0
    h = np.zeros(filter_len)
    for i in range(filter_len):
        if i == mid:
            h[i] = 1.0 - 2.0 * normalized_freq
        else:
            h[i] = (np.sin(np.pi * (i - mid) * (1.0 - 2.0 * normalized_freq)) -
                     np.sin(np.pi * (i - mid) * 2.0 * normalized_freq)) / (np.pi * (i - mid))

    # Apply Keiser window
    window = np.kaiser(filter_len, alpha)
    h *= window
    h /= np.sum(h)

    # Apply FIR filter
    out = np.copy(audio)
    for ch in range(out.shape[0]):
        out[ch] = np.convolve(audio[ch], h, mode='same')
    return out


def parallel_compress(audio, sr, dry_gain_db=0, wet_gain_db=-6, threshold_db=-20, ratio=4):
    """Parallel compression (phaselimiter parallel_compression).
    Blends compressed and uncompressed signals for natural dynamics."""
    dry = audio.copy()
    wet = multiband_compress(audio, sr, [threshold_db], [ratio])

    dry_gain = 10 ** (dry_gain_db / 20.0)
    wet_gain = 10 ** (wet_gain_db / 20.0)

    return dry * dry_gain + wet * wet_gain


def true_peak_limiter(audio, sr, ceiling_db=-0.5, oversample=4):
    """True peak limiter with oversampling (phaselimiter ceiling_mode=true_peak).
    Detects inter-sample peaks that traditional limiters miss."""
    ceiling = 10 ** (ceiling_db / 20.0)

    out = np.copy(audio)
    for ch in range(out.shape[0]):
        # Upsample for true peak detection
        upsampled = signal.resample(audio[ch], len(audio[ch]) * oversample)
        # Find true peaks
        abs_up = np.abs(upsampled)
        # Downsample peak envelope
        peak_env = np.array([np.max(abs_up[i*oversample:(i+1)*oversample])
                           for i in range(len(audio[ch]))])

        # Apply limiting where true peak exceeds ceiling
        mask = peak_env > ceiling
        if np.any(mask):
            gain = np.ones(len(audio[ch]))
            gain[mask] = ceiling / (peak_env[mask] + 1e-10)
            # Smooth gain changes
            smoothed = np.copy(gain)
            for i in range(1, len(smoothed)):
                smoothed[i] = min(smoothed[i], smoothed[i-1] * 0.999 + smoothed[i] * 0.001)
            for i in range(len(smoothed) - 2, -1, -1):
                smoothed[i] = min(smoothed[i], smoothed[i+1] * 0.999 + smoothed[i] * 0.001)
            out[ch] *= smoothed

    return out


def remaster(audio, sr, p):
    """Full mastering pipeline combining traditional DSP with
    phaselimiter-derived algorithms for superior quality."""

    # Stage 0: Rumble removal (phaselimiter CutLowAndHighFreq)
    audio = highpass_fir(audio, sr, cutoff_freq=p.get("highpass_freq", 20))

    # Stage 1: EQ Chain
    # 1a. Low-shelf bass boost
    b,a = design_biquad("low_shelf", p.get("bass_freq",150), sr, p.get("bass_boost_db",6), 0.707)
    audio = apply_iir(audio, b, a)
    # 1b. Sub-bass peak
    b,a = design_biquad("peaking", p.get("sub_bass_freq",60), sr, p.get("sub_bass_boost_db",4), 0.5)
    audio = apply_iir(audio, b, a)
    # 1c. Mid peak
    b,a = design_biquad("peaking", p.get("mid_freq",3000), sr, p.get("mid_db",2), 1.0)
    audio = apply_iir(audio, b, a)
    # 1d. High-shelf
    b,a = design_biquad("high_shelf", p.get("high_freq",8000), sr, p.get("high_db",1.5), 0.707)
    audio = apply_iir(audio, b, a)
    # 1e. Optional custom EQ bands
    for eq in p.get("custom_eq", []):
        b,a = design_biquad(eq["type"], eq["freq"], sr, eq["gain"], eq.get("q",0.707))
        audio = apply_iir(audio, b, a)

    # Stage 2: Loudness mapping compression (phaselimiter LoudnessMapping)
    # More natural than traditional threshold/ratio compression
    audio = loudness_mapping_compress(audio, sr,
        target_lufs=p.get("target_lufs", -14),
        strength=p.get("loudness_mapping_strength", 0.5))

    # Stage 3: Mid/Side compression (phaselimiter MsCompressor)
    audio = ms_compressor(audio, sr,
        threshold_db=p.get("ms_threshold", -22),
        ratio=p.get("ms_ratio", 2.0),
        side_boost_db=p.get("ms_side_boost", 0))

    # Stage 4: Multiband compressor (traditional)
    ct = p.get("comp_threshold", -20)
    cr = p.get("comp_ratio", 3)
    audio = multiband_compress(audio, sr, [ct]*3, [cr]*3)

    # Stage 5: Parallel compression (phaselimiter)
    audio = parallel_compress(audio, sr,
        dry_gain_db=0,
        wet_gain_db=p.get("parallel_wet_db", -6),
        threshold_db=p.get("parallel_threshold", -24),
        ratio=p.get("parallel_ratio", 4))

    # Stage 6: Phase-coherent limiter (phaselimiter core)
    audio = phase_limit(audio, sr,
        ceiling_db=p.get("limiter_ceiling", -0.5),
        lookahead_ms=p.get("lookahead_ms", 5),
        release_ms=p.get("limiter_release_ms", 100))

    # Stage 7: True peak limiter (phaselimiter true_peak)
    audio = true_peak_limiter(audio, sr,
        ceiling_db=p.get("true_peak_ceiling", -0.3),
        oversample=p.get("true_peak_oversample", 4))

    # Stage 8: LUFS normalization
    audio = normalize_lufs(audio, sr, p.get("target_lufs", -14))

    # Stage 9: Stereo enhancement
    sw = p.get("stereo_width", 1.2)
    if sw != 1.0:
        audio = stereo_enhance(audio, sw)

    # Stage 10: Final safety clip
    return np.clip(audio, -1.0, 1.0)


# ── HTML Template ────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>🎛️ Remaster Studio</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#e6edf3;
--dim:#8b949e;--accent:#58a6ff;--accent2:#f0883e;--green:#3fb950;--red:#f85149;
--purple:#bc8cff;--radius:8px}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
min-height:100vh;padding:0}
h1{font-size:1.4rem;font-weight:700}
.header{background:var(--card);border-bottom:1px solid var(--border);padding:12px 24px;
display:flex;align-items:center;gap:16px;position:sticky;top:0;z-index:100}
.header h1{white-space:nowrap}
.header .sub{color:var(--dim);font-size:.8rem}
.layout{display:grid;grid-template-columns:320px 1fr;min-height:calc(100vh - 53px)}
.sidebar{background:var(--card);border-right:1px solid var(--border);padding:16px;
overflow-y:auto;max-height:calc(100vh - 53px)}
.main{padding:16px;overflow-y:auto;max-height:calc(100vh - 53px)}
.section{margin-bottom:16px}
.section-title{font-size:.75rem;text-transform:uppercase;color:var(--dim);
letter-spacing:.08em;margin-bottom:8px;font-weight:600}
.card{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);
padding:12px;margin-bottom:12px}
label{display:block;font-size:.8rem;color:var(--dim);margin-bottom:4px}
input[type=range]{width:100%;accent-color:var(--accent);height:6px;margin:4px 0}
.val{float:right;color:var(--accent);font-weight:600;font-size:.8rem;font-variant-numeric:tabular-nums}
select,input[type=number]{background:var(--bg);color:var(--text);border:1px solid var(--border);
border-radius:4px;padding:6px 8px;width:100%;font-size:.85rem}
.btn{padding:10px 20px;border:none;border-radius:var(--radius);font-size:.85rem;
font-weight:600;cursor:pointer;transition:all .15s}
.btn-primary{background:var(--accent);color:#000}
.btn-primary:hover{background:#79c0ff}
.btn-primary:disabled{opacity:.4;cursor:not-allowed}
.btn-secondary{background:var(--border);color:var(--text)}
.btn-secondary:hover{background:#484f58}
.btn-danger{background:var(--red);color:#fff}
.btn-sm{padding:6px 12px;font-size:.75rem}
.upload-zone{border:2px dashed var(--border);border-radius:var(--radius);padding:40px;
text-align:center;cursor:pointer;transition:all .2s}
.upload-zone:hover,.upload-zone.dragover{border-color:var(--accent);background:rgba(88,166,255,.05)}
.upload-zone input{display:none}
.upload-zone .icon{font-size:2.5rem;margin-bottom:8px}
.upload-zone .hint{color:var(--dim);font-size:.8rem;margin-top:4px}
.file-info{display:flex;align-items:center;gap:12px;padding:12px;
background:var(--card);border:1px solid var(--border);border-radius:var(--radius);margin-bottom:12px}
.file-info .name{font-weight:600;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.file-info .meta{color:var(--dim);font-size:.8rem}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
.eq-band{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
padding:10px;text-align:center}
.eq-band .freq{font-size:.7rem;color:var(--dim)}
.eq-band .gain-val{font-size:1rem;font-weight:700;margin:4px 0}
.eq-band input[type=range]{writing-mode:vertical-lr;height:80px;width:auto;margin:4px auto}
.spectrum-container{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
padding:16px;margin-bottom:12px;position:relative}
.spectrum-container h3{font-size:.85rem;margin-bottom:8px}
canvas{width:100%;height:200px;border-radius:4px}
.band-bars{display:flex;gap:6px;align-items:flex-end;height:120px;padding:8px 0}
.band-bar-wrap{flex:1;text-align:center}
.band-bar{width:100%;background:var(--accent);border-radius:4px 4px 0 0;
transition:height .3s;min-height:2px;position:relative}
.band-bar.after{background:var(--green)}
.band-label{font-size:.6rem;color:var(--dim);margin-top:4px;word-break:keep-all}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px}
.stat{text-align:center;padding:10px;background:var(--card);border:1px solid var(--border);border-radius:var(--radius)}
.stat .num{font-size:1.2rem;font-weight:700;color:var(--accent)}
.stat .unit{font-size:.7rem;color:var(--dim)}
.preset-chips{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:4px}
.chip{padding:5px 10px;border-radius:16px;font-size:.7rem;font-weight:600;
border:1px solid var(--border);cursor:pointer;transition:all .15s;background:var(--bg);color:var(--dim)}
.chip:hover{border-color:var(--accent);color:var(--text)}
.chip.active{background:var(--accent);color:#000;border-color:var(--accent)}
.spinner{display:inline-block;width:16px;height:16px;border:2px solid var(--border);
border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.progress-bar{height:4px;background:var(--border);border-radius:2px;overflow:hidden;margin:8px 0}
.progress-bar .fill{height:100%;background:var(--accent);transition:width .3s;width:0%}
.toast{position:fixed;bottom:20px;right:20px;padding:12px 20px;border-radius:var(--radius);
background:var(--green);color:#000;font-weight:600;font-size:.85rem;opacity:0;
transform:translateY(20px);transition:all .3s;z-index:999}
.toast.show{opacity:1;transform:translateY(0)}
.download-section{text-align:center;padding:32px}
.download-section .big{font-size:3rem;margin-bottom:12px}
.hidden{display:none}
.toggle{position:relative;width:36px;height:20px;background:var(--border);
border-radius:10px;cursor:pointer;transition:.2s}
.toggle.on{background:var(--accent)}
.toggle::after{content:'';position:absolute;width:16px;height:16px;background:#fff;
border-radius:50%;top:2px;left:2px;transition:.2s}
.toggle.on::after{left:18px}
.eq-add{background:none;border:1px dashed var(--border);color:var(--dim);
border-radius:var(--radius);padding:8px;cursor:pointer;font-size:.75rem;width:100%}
.eq-add:hover{border-color:var(--accent);color:var(--accent)}
.tab-bar{display:flex;gap:0;border-bottom:1px solid var(--border);margin-bottom:12px}
.tab{padding:8px 16px;font-size:.8rem;color:var(--dim);cursor:pointer;border-bottom:2px solid transparent}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.comparison{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.comp-side{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:12px}
.comp-side h4{font-size:.8rem;margin-bottom:8px}
.comp-side.before h4{color:var(--accent2)}
.comp-side.after h4{color:var(--green)}
</style>
</head>
<body>
<div class="header">
  <h1>🎛️ Remaster Studio</h1>
  <span class="sub">高音質 重低音重視 リマスター</span>
  <div style="flex:1"></div>
  <span id="statusText" class="sub"></span>
</div>

<div class="layout">
<!-- Sidebar -->
<div class="sidebar" id="sidebar">

  <!-- Upload -->
  <div class="section">
    <div class="section-title">📁 ファイル</div>
    <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
      <div class="icon">🎵</div>
      <div>クリック / ドラッグ&ドロップ</div>
      <div class="hint">WAV, MP3, FLAC, OGG (最大200MB)</div>
      <input type="file" id="fileInput" accept="audio/*">
    </div>
    <div id="fileInfo" class="file-info hidden">
      <span>🎶</span>
      <span class="name" id="fileName"></span>
      <span class="meta" id="fileMeta"></span>
    </div>
  </div>

  <!-- Presets -->
  <div class="section">
    <div class="section-title">🎚️ プリセット</div>
    <div class="preset-chips">
      <div class="chip active" data-preset="default" onclick="loadPreset('default')">⚖️ デフォルト</div>
      <div class="chip" data-preset="hiphop" onclick="loadPreset('hiphop')">🎤 Hip-Hop</div>
      <div class="chip" data-preset="edm" onclick="loadPreset('edm')">🎧 EDM</div>
      <div class="chip" data-preset="pop" onclick="loadPreset('pop')">🎵 Pop</div>
      <div class="chip" data-preset="rock" onclick="loadPreset('rock')">🎸 Rock</div>
      <div class="chip" data-preset="jazz" onclick="loadPreset('jazz')">🎷 Jazz</div>
      <div class="chip" data-preset="classical" onclick="loadPreset('classical')">🎻 古典</div>
      <div class="chip" data-preset="rnb" onclick="loadPreset('rnb')">🎹 R&B</div>
      <div class="chip" data-preset="lofi" onclick="loadPreset('lofi')">📻 Lo-Fi</div>
      <div class="chip" data-preset="bassmusic" onclick="loadPreset('bassmusic')">🔊 Bass</div>
      <div class="chip" data-preset="podcast" onclick="loadPreset('podcast')">🎙️ Podcast</div>
      <div class="chip" data-preset="cinematic" onclick="loadPreset('cinematic')">🎬 Cinematic</div>
      <div class="chip" data-preset="streaming" onclick="loadPreset('streaming')">📺 Streaming</div>
      <div class="chip" data-preset="club" onclick="loadPreset('club')">🏠 Club</div>
      <div class="chip" data-preset="vocal" onclick="loadPreset('vocal')">🎤 Vocal</div>
      <div class="chip" data-preset="audiophile" onclick="loadPreset('audiophile')">🎧 Hi-Fi</div>
      <div class="chip" data-preset="movie" onclick="loadPreset('movie')">🎬 映画</div>
      <div class="chip" data-preset="custom" onclick="loadPreset('custom')">⚙️ Custom</div>
    </div>
    <div id="presetDesc" style="font-size:.75rem;color:var(--accent);margin-top:6px;min-height:1.2em;font-style:italic">バランス型 — 全ジャンル対応</div>
  </div>

  <!-- Bass EQ -->
  <div class="section">
    <div class="section-title">🔊 重低音 EQ</div>
    <div class="card">
      <label>Sub-Bass (20-60Hz) <span class="val" id="subBassVal">+4.0 dB</span></label>
      <input type="range" id="subBass" min="-12" max="18" step="0.5" value="4"
             oninput="updateVal('subBass','subBassVal','dB')">
      <label>Low-Bass (60-200Hz) <span class="val" id="bassVal">+6.0 dB</span></label>
      <input type="range" id="bass" min="-12" max="18" step="0.5" value="6"
             oninput="updateVal('bass','bassVal','dB')">
      <label>Bass Frequency <span class="val" id="bassFreqVal">150 Hz</span></label>
      <input type="range" id="bassFreq" min="40" max="300" step="10" value="150"
             oninput="updateVal('bassFreq','bassFreqVal','Hz')">
      <label>Sub-Bass Frequency <span class="val" id="subFreqVal">60 Hz</span></label>
      <input type="range" id="subFreq" min="20" max="120" step="5" value="60"
             oninput="updateVal('subFreq','subFreqVal','Hz')">
    </div>
  </div>

  <!-- Mid/High EQ -->
  <div class="section">
    <div class="section-title">🎚️ ミッド / ハイ</div>
    <div class="card">
      <label>Mid (クリアリティ) <span class="val" id="midVal">+2.0 dB</span></label>
      <input type="range" id="mid" min="-12" max="12" step="0.5" value="2"
             oninput="updateVal('mid','midVal','dB')">
      <label>Mid Frequency <span class="val" id="midFreqVal">3000 Hz</span></label>
      <input type="range" id="midFreq" min="500" max="8000" step="100" value="3000"
             oninput="updateVal('midFreq','midFreqVal','Hz')">
      <label>High-Shelf (明るさ) <span class="val" id="highVal">+1.5 dB</span></label>
      <input type="range" id="high" min="-6" max="12" step="0.5" value="1.5"
             oninput="updateVal('high','highVal','dB')">
      <label>High Frequency <span class="val" id="highFreqVal">8000 Hz</span></label>
      <input type="range" id="highFreq" min="4000" max="16000" step="500" value="8000"
             oninput="updateVal('highFreq','highFreqVal','Hz')">
    </div>
  </div>

  <!-- Custom EQ Band -->
  <div class="section">
    <div class="section-title">🔧 カスタム EQ</div>
    <div id="customEqList"></div>
    <button class="eq-add" onclick="addEqBand()">+ EQ バンド追加</button>
  </div>

  <!-- Dynamics -->
  <div class="section">
    <div class="section-title">📈 ダイナミクス</div>
    <div class="card">
      <label>コンプ 閾値 <span class="val" id="compThVal">-20 dB</span></label>
      <input type="range" id="compTh" min="-40" max="-6" step="1" value="-20"
             oninput="updateVal('compTh','compThVal','dB')">
      <label>コンプ 比率 <span class="val" id="compRaVal">3.0 : 1</span></label>
      <input type="range" id="compRa" min="1" max="8" step="0.5" value="3"
             oninput="updateVal('compRa','compRaVal',': 1')">
      <label>リミッター上限 <span class="val" id="limVal">-0.5 dBFS</span></label>
      <input type="range" id="lim" min="-3" max="0" step="0.1" value="-0.5"
             oninput="updateVal('lim','limVal','dBFS')">
    </div>
  </div>

  <!-- Output -->
  <div class="section">
    <div class="section-title">📤 出力</div>
    <div class="card">
      <label>ラウドネス目標 <span class="val" id="lufsVal">-14 LUFS</span></label>
      <input type="range" id="lufs" min="-32" max="-6" step="1" value="-14"
             oninput="updateVal('lufs','lufsVal','LUFS')">
      <label>ステレオ幅 <span class="val" id="stereoVal">1.2x</span></label>
      <input type="range" id="stereo" min="0.5" max="2.0" step="0.1" value="1.2"
             oninput="updateVal('stereo','stereoVal','x')">
      <label>サンプルレート</label>
      <select id="sampleRate">
        <option value="44100">44,100 Hz (CD)</option>
        <option value="48000" selected>48,000 Hz</option>
        <option value="96000">96,000 Hz (Hi-Res)</option>
      </select>
      <div style="margin-top:8px">
        <label>出力形式</label>
        <select id="outFormat">
          <option value="wav">WAV (無圧縮)</option>
          <option value="mp3">MP3 (320kbps)</option>
          <option value="flac">FLAC (可逆)</option>
          <option value="ogg">OGG Vorbis</option>
        </select>
      </div>
    </div>
  </div>

  <!-- Actions -->
  <div class="section">
    <button class="btn btn-primary" style="width:100%;margin-bottom:8px" id="btnRemaster"
            onclick="doRemaster()" disabled>
      🎛️ リマスター実行
    </button>
    <button class="btn btn-secondary" style="width:100%" id="btnAnalyze"
            onclick="doAnalyze()" disabled>
      📊 分析のみ
    </button>
  </div>
</div>

<!-- Main -->
<div class="main" id="mainArea">
  <!-- Mode Tabs -->
  <div class="tab-bar" id="modeTabs">
    <div class="tab active" data-mode="single" onclick="switchMode('single')">🎛️ シングル</div>
    <div class="tab" data-mode="batch" onclick="switchMode('batch')">📦 バッチ (複数ファイル)</div>
    <div class="tab" data-mode="ai" onclick="switchMode('ai')">🤖 AI自動最適化</div>
  </div>

  <!-- Welcome -->
  <div id="welcome">
    <div style="text-align:center;padding:60px 20px">
      <div style="font-size:4rem;margin-bottom:16px">🎛️</div>
      <h2 style="margin-bottom:8px">Remaster Studio</h2>
      <p style="color:var(--dim);max-width:400px;margin:0 auto">
        WAV / MP3 / FLAC を高音質にリマスター。<br>
        重低音ブースト、パラメトリックEQ、<br>
        マルチバンドコンプ、LUFS正規化に対応。
      </p>
      <p style="color:var(--dim);margin-top:16px;font-size:.8rem">
        ← 左のパネルからファイルをアップロードしてね
      </p>
    </div>
  </div>

  <!-- Spectrum -->
  <div id="spectrumSection" class="hidden mode-single">
    <div class="stats" id="statsRow"></div>
    <div class="comparison">
      <div class="comp-side before">
        <h4>📥 Before</h4>
        <canvas id="canvasBefore" height="180"></canvas>
      </div>
      <div class="comp-side after" id="afterSide">
        <h4>📤 After</h4>
        <canvas id="canvasAfter" height="180"></canvas>
        <div id="afterPlaceholder" style="text-align:center;color:var(--dim);padding:60px 0;font-size:.85rem">
          リマスター後に表示されます
        </div>
      </div>
    </div>
    <div style="margin-top:12px">
      <div class="section-title">📊 バンド別エネルギー</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        <div class="comp-side before">
          <h4>Before</h4>
          <div class="band-bars" id="barsBefore"></div>
        </div>
        <div class="comp-side after">
          <h4>After</h4>
          <div class="band-bars" id="barsAfter"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Progress -->
  <div id="progressSection" class="hidden" style="text-align:center;padding:40px">
    <div class="spinner" style="width:40px;height:40px;border-width:3px;margin:0 auto 16px"></div>
    <h3 id="progressTitle">リマスター中...</h3>
    <p id="progressDetail" style="color:var(--dim);margin-top:4px;font-size:.85rem"></p>
    <div class="progress-bar" style="max-width:300px;margin:16px auto">
      <div class="fill" id="progressFill"></div>
    </div>
  </div>

  <!-- Download -->
  <div id="downloadSection" class="hidden download-section">
    <div class="big">✅</div>
    <h3 style="margin-bottom:4px">リマスター完了！</h3>
    <p style="color:var(--dim);margin-bottom:20px" id="downloadInfo"></p>
    <a id="downloadLink" class="btn btn-primary" style="text-decoration:none;display:inline-block">
      💾 ダウンロード
    </a>
    <div style="margin-top:12px">
      <button class="btn btn-secondary btn-sm" onclick="resetAll()">🔄 もう一度</button>
    </div>

    <!-- Batch Mode: Upload -->
    <div id="batchUploadSection" class="hidden">
      <div class="upload-zone" id="batchUploadZone" onclick="document.getElementById('batchFileInput').click()" style="margin-bottom:16px">
        <div class="icon">📂</div>
        <div>複数ファイルをドラッグ&ドロップ</div>
        <div class="hint">WAV, MP3, FLAC, OGG — 複数選択OK (最大20ファイル)</div>
        <input type="file" id="batchFileInput" accept="audio/*" multiple style="display:none">
      </div>
      <div id="batchFileList"></div>
      <div id="batchActions" class="hidden" style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">
        <button class="btn btn-primary" onclick="doBatchRemaster()" id="btnBatchRemaster">📦 一括リマスター</button>
        <button class="btn btn-secondary" onclick="resetBatch()">🔄 リセット</button>
      </div>
      <div id="batchResults" class="hidden" style="margin-top:16px"></div>
    </div>

    <!-- AI Auto Mode -->
    <div id="aiSection" class="hidden">
      <div class="upload-zone" id="aiUploadZone" onclick="document.getElementById('aiFileInput').click()" style="margin-bottom:16px">
        <div class="icon">🤖</div>
        <div>ファイルをアップロード → AIが自動分析・最適化</div>
        <div class="hint">AIがジャンルを判定し、最適なマスタリング設定を自動提案します</div>
        <input type="file" id="aiFileInput" accept="audio/*" multiple style="display:none">
      </div>
      <div id="aiFileList"></div>
      <div id="aiResults" class="hidden" style="margin-top:16px"></div>
    </div>

    </div>
    </div>

    <div class="toast" id="toast"></div>

<script>
// State
let fileId = null;
let beforeData = null;
let afterData = null;
let eqBands = [];
let eqIdCounter = 0;
let currentMode = 'single';
let batchFiles = []; // [{file_id, filename, analysis, params}]
let aiFiles = [];    // [{file_id, filename, analysis, recommended}]

// ── Mode Switching ─────────────────────────────────────────────────
function switchMode(mode) {
  currentMode = mode;
  document.querySelectorAll('#modeTabs .tab').forEach(t=>t.classList.remove('active'));
  document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
  // Show/hide sections
  document.getElementById('welcome').classList.toggle('hidden', mode!=='single');
  document.getElementById('spectrumSection').classList.toggle('hidden', mode!=='single');
  document.getElementById('progressSection').classList.add('hidden');
  document.getElementById('downloadSection').classList.add('hidden');
  document.getElementById('batchUploadSection').classList.toggle('hidden', mode!=='batch');
  document.getElementById('aiSection').classList.toggle('hidden', mode!=='ai');
  // Sidebar: show preset controls only in single mode
  document.querySelectorAll('.sidebar .section').forEach(s=>{
    if (s.querySelector('.section-title')?.textContent.includes('プリセット') ||
        s.querySelector('.section-title')?.textContent.includes('重低音') ||
        s.querySelector('.section-title')?.textContent.includes('ミッド') ||
        s.querySelector('.section-title')?.textContent.includes('カスタム') ||
        s.querySelector('.section-title')?.textContent.includes('ダイナミクス') ||
        s.querySelector('.section-title')?.textContent.includes('出力')) {
      s.style.display = mode==='single' ? '' : 'none';
    }
  });
}

// ── Batch Upload ────────────────────────────────────────────────────
const batchUploadZone = document.getElementById('batchUploadZone');
const batchFileInput = document.getElementById('batchFileInput');
batchUploadZone.addEventListener('dragover', e=>{e.preventDefault();batchUploadZone.classList.add('dragover')});
batchUploadZone.addEventListener('dragleave', ()=>batchUploadZone.classList.remove('dragover'));
batchUploadZone.addEventListener('drop', e=>{e.preventDefault();batchUploadZone.classList.remove('dragover');
  if(e.dataTransfer.files.length) batchUploadFiles(e.dataTransfer.files)});
batchFileInput.addEventListener('change', ()=>{if(batchFileInput.files.length) batchUploadFiles(batchFileInput.files)});

async function batchUploadFiles(fileList) {
  const fd = new FormData();
  const names = [];
  for (const f of fileList) { fd.append('files', f); names.push(f.name); }
  setStatus(`${names.length}ファイル アップロード中...`);
  try {
    const r = await fetch('/upload-batch', {method:'POST', body:fd});
    const d = await r.json();
    batchFiles = d.files.filter(f=>!f.error).map(f=>({...f, params:null}));
    renderBatchList();
    document.getElementById('batchActions').classList.remove('hidden');
    toast(`✅ ${batchFiles.length}ファイル アップロード完了`);
  } catch(e) { toast('❌ アップロード失敗'); }
  setStatus('');
}

function renderBatchList() {
  const el = document.getElementById('batchFileList');
  el.innerHTML = batchFiles.map((f,i)=>`
    <div class="card" style="margin-bottom:6px;padding:10px">
      <div style="display:flex;align-items:center;gap:8px">
        <span style="font-size:1.2rem">🎶</span>
        <div style="flex:1;min-width:0">
          <div style="font-weight:600;font-size:.85rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${f.filename}</div>
          <div style="font-size:.7rem;color:var(--dim)">${f.channels} / ${f.sr}Hz / ${f.duration}s</div>
        </div>
        <select class="batch-preset-select" data-idx="${i}" style="width:120px;font-size:.75rem;padding:4px 6px"
                onchange="batchFiles[${i}].params=this.value==='auto'?null:this.value">
          <option value="auto">🤖 自動</option>
          ${Object.keys(PRESETS).filter(k=>k!=='custom').map(k=>`<option value="${k}">${k}</option>`).join('')}
        </select>
      </div>
    </div>
  `).join('');
}

async function doBatchRemaster() {
  if (!batchFiles.length) return;
  document.getElementById('btnBatchRemaster').disabled = true;
  setStatus('一括リマスター中...');
  const items = batchFiles.map(f=>({
    file_id: f.file_id,
    params: f.params || PRESETS.default,
    format: document.getElementById('outFormat')?.value || 'wav',
  }));
  try {
    const r = await fetch('/batch-remaster', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({items})
    });
    const d = await r.json();
    renderBatchResults(d.results);
    toast(`✅ ${d.results.filter(r=>!r.error).length}ファイル リマスター完了`);
  } catch(e) { toast('❌ リマスター失敗'); }
  document.getElementById('btnBatchRemaster').disabled = false;
  setStatus('');
}

function renderBatchResults(results) {
  const el = document.getElementById('batchResults');
  el.classList.remove('hidden');
  el.innerHTML = '<div class="section-title">📥 ダウンロード</div>' + results.map(r=>{
    if (r.error) return `<div class="card" style="padding:8px;color:var(--red)">❌ ${r.filename}: ${r.error}</div>`;
    const sizeMB = (r.size/1024/1024).toFixed(1);
    return `<div class="card" style="padding:8px;display:flex;align-items:center;gap:8px">
      <span>✅</span>
      <span style="flex:1;font-size:.85rem">${r.filename}</span>
      <span style="font-size:.7rem;color:var(--dim)">${sizeMB}MB</span>
      <a href="/download/${r.output}" class="btn btn-primary btn-sm" download style="text-decoration:none;font-size:.75rem">💾 DL</a>
    </div>`;
  }).join('');
}

function resetBatch() {
  batchFiles = [];
  document.getElementById('batchFileList').innerHTML = '';
  document.getElementById('batchActions').classList.add('hidden');
  document.getElementById('batchResults').classList.add('hidden');
  batchFileInput.value = '';
}

// ── AI Auto-Optimize ────────────────────────────────────────────────
const aiUploadZone = document.getElementById('aiUploadZone');
const aiFileInput = document.getElementById('aiFileInput');
aiUploadZone.addEventListener('dragover', e=>{e.preventDefault();aiUploadZone.classList.add('dragover')});
aiUploadZone.addEventListener('dragleave', ()=>aiUploadZone.classList.remove('dragover'));
aiUploadZone.addEventListener('drop', e=>{e.preventDefault();aiUploadZone.classList.remove('dragover');
  if(e.dataTransfer.files.length) aiUploadFiles(e.dataTransfer.files)});
aiFileInput.addEventListener('change', ()=>{if(aiFileInput.files.length) aiUploadFiles(aiFileInput.files)});

async function aiUploadFiles(fileList) {
  const fd = new FormData();
  for (const f of fileList) fd.append('files', f);
  setStatus('アップロード + AI分析中...');
  try {
    // Upload
    const ur = await fetch('/upload-batch', {method:'POST', body:fd});
    const ud = await ur.json();
    const fileIds = ud.files.filter(f=>!f.error).map(f=>f.file_id);
    if (!fileIds.length) { toast('❌ アップロード失敗'); setStatus(''); return; }
    // AI optimize
    const ar = await fetch('/ai-optimize', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({file_ids:fileIds})
    });
    const ad = await ar.json();
    aiFiles = ad.results.filter(r=>!r.error);
    renderAiResults();
    toast(`✅ ${aiFiles.length}ファイル AI分析完了`);
  } catch(e) { toast('❌ AI分析失敗: '+e.message); }
  setStatus('');
}

function renderAiResults() {
  const el = document.getElementById('aiResults');
  el.classList.remove('hidden');
  el.innerHTML = aiFiles.map(f=>`
    <div class="card" style="margin-bottom:12px;padding:16px">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
        <span style="font-size:1.5rem">🎵</span>
        <div style="flex:1">
          <div style="font-weight:700;font-size:1rem">${f.filename}</div>
          <div style="font-size:.75rem;color:var(--dim)">${f.analysis_summary.duration}s / ${f.analysis_summary.rms_db} dBFS RMS</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:.85rem;font-weight:600;color:var(--accent)">${genreLabel(f.genre)}</div>
          <div style="font-size:.7rem;color:var(--dim)">信頼度 ${(f.confidence*100).toFixed(0)}%</div>
        </div>
      </div>
      <div style="background:var(--bg);border-radius:var(--radius);padding:10px;margin-bottom:8px">
        <div class="section-title" style="margin-bottom:6px">🤖 AI推奨設定</div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;font-size:.75rem">
          <div><span style="color:var(--dim)">Sub-Bass:</span> <b>${f.settings.subBass>0?'+':''}${f.settings.subBass}dB</b></div>
          <div><span style="color:var(--dim)">Bass:</span> <b>${f.settings.bass>0?'+':''}${f.settings.bass}dB</b></div>
          <div><span style="color:var(--dim)">Mid:</span> <b>${f.settings.mid>0?'+':''}${f.settings.mid}dB</b></div>
          <div><span style="color:var(--dim)">High:</span> <b>${f.settings.high>0?'+':''}${f.settings.high}dB</b></div>
          <div><span style="color:var(--dim)">Comp:</span> <b>${f.settings.compTh}dB / ${f.settings.compRa}:1</b></div>
          <div><span style="color:var(--dim)">Limiter:</span> <b>${f.settings.lim}dBFS</b></div>
          <div><span style="color:var(--dim)">LUFS:</span> <b>${f.settings.lufs}</b></div>
          <div><span style="color:var(--dim)">Stereo:</span> <b>${f.settings.stereo}x</b></div>
        </div>
      </div>
      <div style="font-size:.75rem;color:var(--dim);margin-bottom:8px">
        <b>💬 AI判断理由:</b><br>
        ${f.reasoning.map(r=>`• ${r}`).join('<br>')}
      </div>
      <div style="display:flex;gap:6px">
        <button class="btn btn-primary btn-sm" onclick="aiApplyAndDownload('${f.file_id}',this)">💾 この設定でDL</button>
        <button class="btn btn-secondary btn-sm" onclick="aiApplyToSingle('${f.file_id}','${f.genre}')">🎛️ シングルモードで開く</button>
      </div>
    </div>
  `).join('');
}

function genreLabel(g) {
  const labels = {hiphop:'🎤 Hip-Hop',edm:'🎧 EDM',pop:'🎵 Pop',rock:'🎸 Rock',jazz:'🎷 Jazz',
    classical:'🎻 古典',rnb:'🎹 R&B',lofi:'📻 Lo-Fi',bassmusic:'🔊 Bass Music',
    podcast:'🎙️ Podcast',cinematic:'🎬 Cinematic',default:'⚖️ デフォルト'};
  return labels[g]||g;
}

async function aiApplyAndDownload(fileId, btn) {
  btn.disabled = true; btn.textContent = '⏳ 処理中...';
  const aiResult = aiFiles.find(f=>f.file_id===fileId);
  if (!aiResult) return;
  const fmt = document.getElementById('outFormat')?.value || 'wav';
  try {
    const r = await fetch('/remaster', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({file_id:fileId, params:aiResult.settings, format:fmt})
    });
    if (!r.ok) throw new Error('remaster failed');
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `remastered_${aiResult.filename}.${fmt}`;
    a.click(); URL.revokeObjectURL(url);
    btn.textContent = '✅ 完了';
  } catch(e) { btn.textContent = '❌ エラー'; }
  setTimeout(()=>{btn.disabled=false;btn.textContent='💾 この設定でDL'},2000);
}

function aiApplyToSingle(fileId, genre) {
  // Switch to single mode and load the AI-recommended preset
  switchMode('single');
  const aiResult = aiFiles.find(f=>f.file_id===fileId);
  if (aiResult && PRESETS[genre]) {
    loadPreset(genre);
    toast(`🎛️ ${genreLabel(genre)} プリセットを適用しました`);
  }
}

// ── Professional Mastering Presets ──────────────────────────────────
// Based on industry-standard chains: iZotope Ozone, FabFilter Pro-MB,
// Waves SSL, Soundtoys, and real mastering engineer workflows.
const PRESETS = {
  // ★ Default — Balanced, all-purpose mastering (Ozone "Clear" reference)
  default: {subBass:4,bass:6,bassFreq:150,subFreq:60,mid:2,midFreq:3000,high:1.5,highFreq:8000,compTh:-20,compRa:3,lim:-0.5,lufs:-14,stereo:1.2,sampleRate:48000,
    desc:'バランス型 — 全ジャンル対応'},

  // ★ Hip-Hop / Trap — 808重視、スクープMID、ブライトハイ
  // Reference: Metro Boomin / Murda Beatz mastering chain
  // Sub-bass boosted hard at 40-60Hz, 200-400Hz scooped for clarity,
  // 3-5kHz presence boost, heavy limiting for loudness
  hiphop: {subBass:10,bass:8,bassFreq:120,subFreq:45,mid:-2,midFreq:300,high:3,highFreq:4000,
    compTh:-16,compRa:5,lim:-0.3,lufs:-9,stereo:1.15,sampleRate:48000,
    desc:'808重視 — 重低音ドリブン'},

  // ★ EDM / Dance — Massive sub, punchy mids, aggressive multiband
  // Reference: Deadmau5 / Eric Prydz mastering approach
  // Sub-bass at 30-50Hz, 1-3kHz punch, 8-12kHz air,
  // Multiband compression tight, loud target
  edm: {subBass:12,bass:10,bassFreq:100,subFreq:40,mid:1,midFreq:2000,high:2,highFreq:10000,
    compTh:-14,compRa:6,lim:-0.2,lufs:-8,stereo:1.4,sampleRate:48000,
    desc:'ダンスフロア向け — 最大級の重低音'},

  // ★ Pop — Radio-ready, present, polished
  // Reference: Max Martin / Shellback production style
  // Gentle sub, 200Hz warmth, 3-4kHz vocal presence,
  // 10kHz+ air, moderate compression for movement
  pop: {subBass:3,bass:4,bassFreq:120,subFreq:60,mid:3,midFreq:3500,high:2.5,highFreq:10000,
    compTh:-18,compRa:3,lim:-0.5,lufs:-12,stereo:1.2,sampleRate:48000,
    desc:'ラジオ・チャート向け — ポリッシュ'},

  // ★ Rock / Metal — Crunchy, aggressive, wide
  // Reference: Andy Wallace / Chris Lord-Alge
  // Low-mid body at 80-150Hz, 1-3kHz guitar aggression,
  // 6-8kHz sizzle, parallel compression for power
  rock: {subBass:5,bass:7,bassFreq:100,subFreq:50,mid:3,midFreq:2500,high:2,highFreq:6000,
    compTh:-16,compRa:4,lim:-0.5,lufs:-10,stereo:1.3,sampleRate:48000,
    desc:'アグレッシブ — ギター・ドラム'},

  // ★ Jazz / Acoustic — Natural, dynamic, warm
  // Reference: Blue Note Records / ECM mastering
  // Subtle bass warmth, natural mids, gentle highs,
  // Minimal compression for dynamics preservation
  jazz: {subBass:2,bass:3,bassFreq:120,subFreq:60,mid:1,midFreq:2000,high:1,highFreq:6000,
    compTh:-24,compRa:2,lim:-1,lufs:-16,stereo:1.15,sampleRate:96000,
    desc:'ナチュラル — ダイナミクス重視'},

  // ★ Classical — Transparent, minimal processing
  // Reference: Decca Records / DG mastering standard
  // Near-flat EQ, very gentle compression,
  // Wide stereo, high sample rate, -24 LUFS streaming
  classical: {subBass:1,bass:1,bassFreq:80,subFreq:40,mid:0,midFreq:1000,high:0.5,highFreq:12000,
    compTh:-30,compRa:1.5,lim:-2,lufs:-24,stereo:1.5,sampleRate:96000,
    desc:'透明 — 最小限の処理'},

  // ★ R&B / Soul — Warm, smooth, vintage character
  // Reference: D'Angelo / Erykah Badu production
  // Warm low-mids at 100-200Hz, smooth 2-4kHz,
  // Gentle highs, moderate compression for groove
  rnb: {subBass:4,bass:5,bassFreq:130,subFreq:55,mid:1,midFreq:2500,high:1.5,highFreq:7000,
    compTh:-18,compRa:3,lim:-0.5,lufs:-12,stereo:1.15,sampleRate:48000,
    desc:'スムーズ — ウォームな質感'},

  // ★ Lo-Fi / Vintage — Character, rolled-off, saturated feel
  // Reference: J Dilla / Nujabes aesthetic
  // Boosted low-mids for warmth, rolled-off highs at 6kHz,
  // Mild compression, gentle limiting
  lofi: {subBass:5,bass:6,bassFreq:150,subFreq:70,mid:2,midFreq:500,high:-1,highFreq:4000,
    compTh:-16,compRa:3,lim:-1,lufs:-14,stereo:1.0,sampleRate:44100,
    desc:'レトロ — 意図的なローファイ'},

  // ★ Bass Music / Dubstep — Extreme sub-bass focus
  // Reference: Skrillex / Excision mastering chain
  // Maximum sub-bass at 30-60Hz, 100-200Hz tight,
  // Aggressive multiband, heavy limiting
  bassmusic: {subBass:15,bass:12,bassFreq:80,subFreq:35,mid:-1,midFreq:2000,high:2,highFreq:8000,
    compTh:-12,compRa:7,lim:-0.1,lufs:-7,stereo:1.3,sampleRate:48000,
    desc:'極限重低音 — サブベース最強'},

  // ★ Podcast / Voice — Speech clarity, de-ess
  // Reference: NPR / Joe Rogan mastering
  // Sub cut at 80Hz, 2-5kHz speech clarity,
  // De-ess at 6-8kHz, heavy compression for consistency
  podcast: {subBass:-2,bass:1,bassFreq:100,subFreq:80,mid:4,midFreq:3000,high:2,highFreq:5000,
    compTh:-20,compRa:4,lim:-0.5,lufs:-16,stereo:1.0,sampleRate:48000,
    desc:'音声クリア — ポッドキャスト'},

  // ★ Cinematic — Epic, wide, dramatic
  // Reference: Hans Zimmer / film trailer mastering
  // Deep sub-bass, wide stereo, moderate compression,
  // Low LUFS for dynamic range
  cinematic: {subBass:8,bass:6,bassFreq:80,subFreq:30,mid:2,midFreq:1500,high:2,highFreq:12000,
    compTh:-22,compRa:2.5,lim:-1,lufs:-20,stereo:1.5,sampleRate:48000,
    desc:'エピック — 映画・予告編'},

  // ★ Loud (Streaming/YouTube) — Optimized for streaming platforms
  // Reference: LUFS normalization standards (-14 Spotify, -16 Apple)
  // Balanced EQ, moderate compression, safe limiting
  streaming: {subBass:3,bass:4,bassFreq:130,subFreq:55,mid:2,midFreq:3000,high:1.5,highFreq:8000,
    compTh:-18,compRa:3.5,lim:-0.5,lufs:-14,stereo:1.2,sampleRate:48000,
    desc:'ストリーミング最適化'},

  // ★ Club / DJ — Maximum loudness, heavy bass
  // Reference: Club sound system calibration
  // Hard sub-bass, scooped mids, aggressive limiting,
  // Target -8 to -10 LUFS for club play
  club: {subBass:10,bass:12,bassFreq:100,subFreq:40,mid:-1,midFreq:2000,high:1,highFreq:8000,
    compTh:-14,compRa:5,lim:-0.2,lufs:-9,stereo:1.3,sampleRate:48000,
    desc:'クラブDJ — 最大ラウドネス'},

  // ★ Vocal / 歌声 — Vocal-forward mastering
  // Reference: Adele / Beyonce vocal chain
  // Sub cut, 200Hz warmth, 3-5kHz presence,
  // 10kHz+ air, moderate compression
  vocal: {subBass:0,bass:3,bassFreq:150,subFreq:80,mid:4,midFreq:3500,high:3,highFreq:10000,
    compTh:-20,compRa:3,lim:-0.5,lufs:-14,stereo:1.0,sampleRate:48000,
    desc:'ボーカルフロント — 歌声メイン'},

  // ★ Hi-Fi / Audiophile — Pristine, dynamic
  // Reference: Reference recording standards
  // Gentle everything, minimal compression,
  // High sample rate, preserve dynamics
  audiophile: {subBass:2,bass:3,bassFreq:150,subFreq:60,mid:0.5,midFreq:3000,high:1,highFreq:12000,
    compTh:-26,compRa:1.8,lim:-1,lufs:-16,stereo:1.2,sampleRate:96000,
    desc:'究極の音質 — オーディオファイル'},

  // ★ Movie / 映画 — Soundtrack mastering
  // Reference: Dolby Atmos / cinema standard
  // Deep sub, wide stereo, moderate compression
  movie: {subBass:6,bass:5,bassFreq:100,subFreq:35,mid:2,midFreq:2000,high:2,highFreq:10000,
    compTh:-24,compRa:2.5,lim:-1,lufs:-24,stereo:1.5,sampleRate:48000,
    desc:'サウンドトラック — 映画・ドラマ'},

  // ★ Custom — User-defined (no defaults)
  custom: null,
};

function updateVal(sliderId, valId, unit) {
  const v = parseFloat(document.getElementById(sliderId).value);
  const el = document.getElementById(valId);
  if (unit==='dB'||unit==='dBFS'||unit==='LUFS') el.textContent = (v>=0?'+':'')+v+' '+unit;
  else if (unit===': 1') el.textContent = v.toFixed(1)+' '+unit;
  else if (unit==='x') el.textContent = v.toFixed(1)+'x';
  else el.textContent = v+' '+unit;
}

function loadPreset(name) {
  document.querySelectorAll('.chip').forEach(c=>c.classList.remove('active'));
  document.querySelector(`[data-preset="${name}"]`).classList.add('active');
  const p = PRESETS[name];
  // Show preset description
  const descEl = document.getElementById('presetDesc');
  if (descEl) descEl.textContent = p ? (p.desc || '') : 'カスタム設定';
  if (!p) return;
  document.getElementById('subBass').value = p.subBass;
  document.getElementById('bass').value = p.bass;
  document.getElementById('bassFreq').value = p.bassFreq;
  document.getElementById('subFreq').value = p.subFreq;
  document.getElementById('mid').value = p.mid;
  document.getElementById('midFreq').value = p.midFreq;
  document.getElementById('high').value = p.high;
  document.getElementById('highFreq').value = p.highFreq;
  document.getElementById('compTh').value = p.compTh;
  document.getElementById('compRa').value = p.compRa;
  document.getElementById('lim').value = p.lim;
  document.getElementById('lufs').value = p.lufs;
  document.getElementById('stereo').value = p.stereo;
  document.getElementById('sampleRate').value = p.sampleRate;
  ['subBass','bass','bassFreq','subFreq','mid','midFreq','high','highFreq','compTh','compRa','lim','lufs','stereo'].forEach(id=>{
    const sl = document.getElementById(id);
    sl.dispatchEvent(new Event('input'));
  });
}

function addEqBand() {
  const id = eqIdCounter++;
  eqBands.push({id, type:'peaking', freq:1000, gain:0, q:0.707});
  renderEqBands();
}

function removeEqBand(id) {
  eqBands = eqBands.filter(b=>b.id!==id);
  renderEqBands();
}

function renderEqBands() {
  const el = document.getElementById('customEqList');
  el.innerHTML = eqBands.map(b=>`
    <div class="card" style="margin-bottom:6px">
      <div style="display:flex;gap:6px;align-items:center;margin-bottom:6px">
        <select style="flex:1" onchange="getEqBand(${b.id}).type=this.value">
          <option value="peaking" ${b.type==='peaking'?'selected':''}>Peak</option>
          <option value="low_shelf" ${b.type==='low_shelf'?'selected':''}>Low Shelf</option>
          <option value="high_shelf" ${b.type==='high_shelf'?'selected':''}>High Shelf</option>
        </select>
        <button class="btn btn-danger btn-sm" onclick="removeEqBand(${b.id})" style="padding:4px 8px">✕</button>
      </div>
      <label>Freq <span class="val">${b.freq} Hz</span></label>
      <input type="range" min="20" max="20000" step="10" value="${b.freq}"
             oninput="getEqBand(${b.id}).freq=+this.value;this.previousElementSibling.querySelector('.val').textContent=this.value+' Hz'">
      <label>Gain <span class="val">${b.gain>=0?'+':''}${b.gain} dB</span></label>
      <input type="range" min="-12" max="12" step="0.5" value="${b.gain}"
             oninput="getEqBand(${b.id}).gain=+this.value;this.previousElementSibling.querySelector('.val').textContent=(this.value>=0?'+':'')+this.value+' dB'">
      <label>Q <span class="val">${b.q}</span></label>
      <input type="range" min="0.1" max="5" step="0.1" value="${b.q}"
             oninput="getEqBand(${b.id}).q=+this.value;this.previousElementSibling.querySelector('.val').textContent=this.value">
    </div>
  `).join('');
}

function getEqBand(id) { return eqBands.find(b=>b.id===id); }

function getParams() {
  return {
    sub_bass_boost_db: parseFloat(document.getElementById('subBass').value),
    bass_boost_db: parseFloat(document.getElementById('bass').value),
    bass_freq: parseFloat(document.getElementById('bassFreq').value),
    sub_bass_freq: parseFloat(document.getElementById('subFreq').value),
    mid_db: parseFloat(document.getElementById('mid').value),
    mid_freq: parseFloat(document.getElementById('midFreq').value),
    high_db: parseFloat(document.getElementById('high').value),
    high_freq: parseFloat(document.getElementById('highFreq').value),
    comp_threshold: parseFloat(document.getElementById('compTh').value),
    comp_ratio: parseFloat(document.getElementById('compRa').value),
    limiter_ceiling: parseFloat(document.getElementById('lim').value),
    target_lufs: parseFloat(document.getElementById('lufs').value),
    stereo_width: parseFloat(document.getElementById('stereo').value),
    sample_rate: parseInt(document.getElementById('sampleRate').value),
    custom_eq: eqBands.map(b=>({type:b.type,freq:b.freq,gain:b.gain,q:b.q})),
  };
}

// Upload
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
uploadZone.addEventListener('dragover', e=>{e.preventDefault();uploadZone.classList.add('dragover')});
uploadZone.addEventListener('dragleave', ()=>uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e=>{e.preventDefault();uploadZone.classList.remove('dragover');
  if(e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0])});
fileInput.addEventListener('change', ()=>{if(fileInput.files.length) uploadFile(fileInput.files[0])});

async function uploadFile(file) {
  setStatus('アップロード中...');
  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch('/upload', {method:'POST', body:fd});
    const d = await r.json();
    if (d.error) { toast('❌ '+d.error); setStatus(''); return; }
    fileId = d.file_id;
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileMeta').textContent = `${d.channels}ch / ${d.sr}Hz / ${d.duration}s`;
    document.getElementById('fileInfo').classList.remove('hidden');
    uploadZone.style.display = 'none';
    document.getElementById('btnRemaster').disabled = false;
    document.getElementById('btnAnalyze').disabled = false;
    beforeData = d.analysis;
    renderBefore();
    document.getElementById('spectrumSection').classList.remove('hidden');
    document.getElementById('welcome').classList.add('hidden');
    toast('✅ アップロード完了');
    setStatus('');
  } catch(e) { toast('❌ アップロード失敗'); setStatus(''); }
}

function renderBefore() {
  if (!beforeData) return;
  // Stats
  const s = beforeData;
  document.getElementById('statsRow').innerHTML = `
    <div class="stat"><div class="num">${s.rms_db}</div><div class="unit">RMS dBFS</div></div>
    <div class="stat"><div class="num">${s.peak_db}</div><div class="unit">Peak dBFS</div></div>
    <div class="stat"><div class="num">${s.crest_db}</div><div class="unit">Crest dB</div></div>
    <div class="stat"><div class="num">${s.duration}s</div><div class="unit">${s.channels}</div></div>
  `;
  // Spectrum
  drawSpectrum('canvasBefore', s.spectrum);
  // Band bars
  renderBars('barsBefore', s.bands, false);
}

function renderBars(id, bands, isAfter) {
  const keys = ['sub_bass_20_60','bass_60_200','low_mid_200_500','mid_500_2k','upper_mid_2k_6k','high_6k_12k','air_12k_20k'];
  const labels = ['Sub','Bass','Low-M','Mid','Up-M','High','Air'];
  const maxH = 100;
  document.getElementById(id).innerHTML = keys.map((k,i)=>{
    const v = bands[k];
    const h = Math.max(2, Math.min(maxH, (v+100)*maxH/80));
    return `<div class="band-bar-wrap">
      <div class="band-bar ${isAfter?'after':''}" style="height:${h}px"></div>
      <div class="band-label">${labels[i]}<br>${v.toFixed(0)}dB</div>
    </div>`;
  }).join('');
}

function drawSpectrum(canvasId, spec) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth * 2;
  canvas.height = canvas.offsetHeight * 2;
  ctx.scale(2,2);
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  ctx.clearRect(0,0,W,H);

  const freqs = spec.freqs;
  const power = spec.power;
  if (!freqs.length) return;

  // Log scale mapping
  const fMin = 20, fMax = 20000;
  const logMin = Math.log10(fMin), logMax = Math.log10(fMax);

  ctx.strokeStyle = canvasId==='canvasBefore' ? '#58a6ff' : '#3fb950';
  ctx.lineWidth = 1.5;
  ctx.beginPath();

  const pMin = -80, pMax = 10;
  let started = false;

  for (let i=0; i<freqs.length; i++) {
    const f = freqs[i];
    if (f < fMin || f > fMax) continue;
    const x = ((Math.log10(f)-logMin)/(logMax-logMin)) * W;
    const y = H - ((power[i]-pMin)/(pMax-pMin)) * H;
    if (!started) { ctx.moveTo(x,y); started = true; }
    else ctx.lineTo(x,y);
  }
  ctx.stroke();

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = 0.5;
  [100,1000,10000].forEach(f=>{
    const x = ((Math.log10(f)-logMin)/(logMax-logMin))*W;
    ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke();
    ctx.fillStyle='rgba(255,255,255,0.3)';
    ctx.font='10px sans-serif';
    ctx.fillText(f>=1000?(f/1000)+'k':f, x+2, H-4);
  });
}

// Remaster
async function doRemaster() {
  if (!fileId) return;
  document.getElementById('progressSection').classList.remove('hidden');
  document.getElementById('spectrumSection').classList.add('hidden');
  document.getElementById('downloadSection').classList.add('hidden');
  document.getElementById('btnRemaster').disabled = true;

  const steps = ['EQ適用中...','重低音ブースト...','コンプレッサー...','リミッター...','LUFS正規化...','書き出し中...'];
  let step = 0;
  const iv = setInterval(()=>{
    step = Math.min(step+1, steps.length-1);
    document.getElementById('progressDetail').textContent = steps[step];
    document.getElementById('progressFill').style.width = ((step+1)/steps.length*100)+'%';
  }, 800);

  try {
    const fmt = document.getElementById('outFormat').value;
    const r = await fetch('/remaster', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({file_id:fileId, params:getParams(), format:fmt})
    });
    clearInterval(iv);
    if (!r.ok) { const e=await r.json(); throw new Error(e.error||'Error'); }
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);

    // Also fetch after-analysis
    const ar = await fetch('/analyze-result', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({file_id:fileId, params:getParams()})
    });
    if (ar.ok) {
      afterData = await ar.json();
      document.getElementById('afterPlaceholder').style.display='none';
      drawSpectrum('canvasAfter', afterData.spectrum);
      renderBars('barsAfter', afterData.bands, true);
    }

    document.getElementById('downloadLink').href = url;
    document.getElementById('downloadLink').download = `remastered.${fmt}`;
    document.getElementById('downloadInfo').textContent =
      `RMS: ${afterData?.rms_db||'?'} dBFS | Peak: ${afterData?.peak_db||'?'} dBFS`;
    document.getElementById('downloadSection').classList.remove('hidden');
    document.getElementById('progressSection').classList.add('hidden');
    document.getElementById('spectrumSection').classList.remove('hidden');
    document.getElementById('btnRemaster').disabled = false;
    toast('✅ 完了！ nya~');
  } catch(e) {
    clearInterval(iv);
    document.getElementById('progressSection').classList.add('hidden');
    document.getElementById('btnRemaster').disabled = false;
    toast('❌ '+e.message);
  }
}

// Analyze only
async function doAnalyze() {
  if (!fileId) return;
  try {
    const r = await fetch('/analyze', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({file_id:fileId})
    });
    const d = await r.json();
    beforeData = d;
    renderBefore();
    document.getElementById('spectrumSection').classList.remove('hidden');
    toast('📊 分析完了');
  } catch(e) { toast('❌ 分析失敗'); }
}

function resetAll() {
  fileId = null; beforeData = null; afterData = null;
  document.getElementById('fileInfo').classList.add('hidden');
  document.getElementById('spectrumSection').classList.add('hidden');
  document.getElementById('downloadSection').classList.add('hidden');
  document.getElementById('progressSection').classList.add('hidden');
  document.getElementById('welcome').classList.remove('hidden');
  document.getElementById('uploadZone').style.display = '';
  document.getElementById('btnRemaster').disabled = true;
  document.getElementById('btnAnalyze').disabled = true;
  document.getElementById('afterPlaceholder').style.display = '';
  fileInput.value = '';
}

function setStatus(t) { document.getElementById('statusText').textContent = t; }
function toast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg; el.classList.add('show');
  setTimeout(()=>el.classList.remove('show'), 3000);
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "ファイルがありません"}), 400
    fid = uuid.uuid4().hex
    ext = Path(f.filename).suffix.lower()
    save_path = UPLOAD_DIR / f"{fid}{ext}"
    f.save(str(save_path))
    try:
        sr = int(request.form.get("sample_rate", 48000))
        audio, asr = read_audio(str(save_path), sr)
        analysis = analyze_spectrum(audio, asr)
        return jsonify({
            "file_id": fid,
            "sr": asr,
            "channels": "stereo" if audio.shape[0]==2 else "mono",
            "duration": analysis["duration"],
            "analysis": analysis,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    fid = data.get("file_id")
    fpath = list(UPLOAD_DIR.glob(f"{fid}.*"))
    if not fpath:
        return jsonify({"error": "ファイルが見つかりません"}), 404
    audio, sr = read_audio(str(fpath[0]))
    return jsonify(analyze_spectrum(audio, sr))


@app.route("/remaster", methods=["POST"])
def remaster_endpoint():
    data = request.json
    fid = data.get("file_id")
    params = data.get("params", {})
    fmt = data.get("format", "wav")
    fpath = list(UPLOAD_DIR.glob(f"{fid}.*"))
    if not fpath:
        return jsonify({"error": "ファイルが見つかりません"}), 404
    try:
        audio, sr = read_audio(str(fpath[0]), params.get("sample_rate", 48000))
        result = remaster(audio, sr, params)
        out_path = OUTPUT_DIR / f"{fid}_remastered.{fmt}"
        write_audio(result, sr, str(out_path), fmt)
        return send_file(str(out_path), as_attachment=True,
                        download_name=f"remastered.{fmt}")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-result", methods=["POST"])
def analyze_result():
    data = request.json
    fid = data.get("file_id")
    params = data.get("params", {})
    fpath = list(UPLOAD_DIR.glob(f"{fid}.*"))
    if not fpath:
        return jsonify({"error": "not found"}), 404
    audio, sr = read_audio(str(fpath[0]), params.get("sample_rate", 48000))
    result = remaster(audio, sr, params)
    return jsonify(analyze_spectrum(result, sr))


@app.route("/upload-batch", methods=["POST"])
def upload_batch():
    """Upload multiple files at once. Returns list of file_ids + analyses."""
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "ファイルがありません"}), 400
    results = []
    for f in files:
        if not f.filename:
            continue
        fid = uuid.uuid4().hex
        ext = Path(f.filename).suffix.lower()
        save_path = UPLOAD_DIR / f"{fid}{ext}"
        f.save(str(save_path))
        try:
            sr = int(request.form.get("sample_rate", 48000))
            audio, asr = read_audio(str(save_path), sr)
            analysis = analyze_spectrum(audio, asr)
            results.append({
                "file_id": fid,
                "filename": f.filename,
                "sr": asr,
                "channels": "stereo" if audio.shape[0]==2 else "mono",
                "duration": analysis["duration"],
                "analysis": analysis,
            })
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})
    return jsonify({"files": results, "count": len(results)})


@app.route("/ai-optimize", methods=["POST"])
def ai_optimize():
    """Analyze file(s) and return AI-recommended mastering settings."""
    data = request.json
    file_ids = data.get("file_ids", [])
    if not file_ids:
        # Single file mode
        fid = data.get("file_id")
        if fid:
            file_ids = [fid]
    if not file_ids:
        return jsonify({"error": "file_id が指定されていません"}), 400

    results = []
    for fid in file_ids:
        fpath = list(UPLOAD_DIR.glob(f"{fid}.*"))
        if not fpath:
            results.append({"file_id": fid, "error": "ファイルが見つかりません"})
            continue
        try:
            audio, sr = read_audio(str(fpath[0]))
            analysis = analyze_spectrum(audio, sr)
            genre, confidence = classify_genre(audio, sr, analysis)
            recommended = recommend_settings(analysis, genre, confidence)
            reasoning = recommended.pop("_reasoning", [])
            results.append({
                "file_id": fid,
                "filename": fpath[0].stem,
                "genre": genre,
                "confidence": confidence,
                "settings": recommended,
                "reasoning": reasoning,
                "analysis_summary": {
                    "rms_db": analysis["rms_db"],
                    "peak_db": analysis["peak_db"],
                    "crest_db": analysis["crest_db"],
                    "duration": analysis["duration"],
                    "bands": analysis["bands"],
                },
            })
        except Exception as e:
            results.append({"file_id": fid, "error": str(e)})
    return jsonify({"results": results, "count": len(results)})


@app.route("/batch-remaster", methods=["POST"])
def batch_remaster():
    """Remaster multiple files with individual or shared settings."""
    data = request.json
    items = data.get("items", [])
    # items = [{file_id, params, format}, ...]
    if not items:
        return jsonify({"error": "items が指定されていません"}), 400

    results = []
    for item in items:
        fid = item.get("file_id")
        params = item.get("params", {})
        fmt = item.get("format", "wav")
        fpath = list(UPLOAD_DIR.glob(f"{fid}.*"))
        if not fpath:
            results.append({"file_id": fid, "error": "ファイルが見つかりません"})
            continue
        try:
            audio, sr = read_audio(str(fpath[0]), params.get("sample_rate", 48000))
            result = remaster(audio, sr, params)
            out_path = OUTPUT_DIR / f"{fid}_remastered.{fmt}"
            write_audio(result, sr, str(out_path), fmt)
            results.append({
                "file_id": fid,
                "filename": fpath[0].stem,
                "output": f"{fid}_remastered.{fmt}",
                "size": out_path.stat().st_size,
            })
        except Exception as e:
            results.append({"file_id": fid, "error": str(e)})
    return jsonify({"results": results, "count": len(results)})


@app.route("/download/<path:filename>")
def download_file(filename):
    """Download a remastered file."""
    fpath = OUTPUT_DIR / filename
    if not fpath.exists():
        return jsonify({"error": "ファイルが見つかりません"}), 404
    return send_file(str(fpath), as_attachment=True, download_name=filename)


if __name__ == "__main__":
    print("🎛️  Remaster Studio 起動中...")
    print("   http://localhost:7860")
    app.run(host="0.0.0.0", port=7860, debug=False)
