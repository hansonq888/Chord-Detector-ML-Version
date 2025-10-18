import streamlit as st
import threading
import time
import numpy as np
from live_chord_detector import LiveChordDetector

# -----------------------------------
# SETUP PAGE
# -----------------------------------
st.set_page_config(page_title="🎵 Live Chord Detector", layout="centered")

st.title("🎵 Live Chord Detector")
st.markdown("Play a chord on your instrument — the model will detect it in real time.")

with st.expander("ℹ️ About", expanded=True):
    st.markdown("""
    Detects 24 chords (C-B major/minor) from any instrument in real-time.
    Uses machine learning to analyze audio features and show confidence levels.
    Only displays chords with >50% confidence for accuracy.
    """)

# -----------------------------------
# APP STATE VARIABLES
# -----------------------------------
st.session_state.setdefault("detector", None)
st.session_state.setdefault("is_detecting", False)
st.session_state.setdefault("thread", None)

# -----------------------------------
# DETECTION LOOP
# -----------------------------------
def detection_loop(detector):
    detector.start_recording()

# -----------------------------------
# CONTROL FUNCTIONS
# -----------------------------------
def start_detection():
    if not st.session_state.is_detecting:
        try:
            detector = LiveChordDetector()
            st.session_state.detector = detector
            st.session_state.is_detecting = True
            thread = threading.Thread(target=detection_loop, args=(detector,), daemon=True)
            st.session_state.thread = thread
            thread.start()
        except Exception as e:
            st.error(f"Error starting detection: {e}")

def stop_detection():
    if st.session_state.is_detecting:
        st.session_state.is_detecting = False
        detector = st.session_state.detector
        if detector:
            detector.stop_recording()
        st.session_state.detector = None
        st.session_state.thread = None

# -----------------------------------
# UI CONTROLS
# -----------------------------------
col1, col2 = st.columns(2)
with col1:
    if st.button("🎙️ Start Detection", disabled=st.session_state.is_detecting):
        start_detection()
with col2:
    if st.button("🛑 Stop Detection", disabled=not st.session_state.is_detecting):
        stop_detection()

st.divider()

# -----------------------------------
# DISPLAY SECTION
# -----------------------------------
placeholder_chord = st.empty()
placeholder_conf = st.empty()
placeholder_bar = st.empty()
placeholder_history = st.empty()

# -----------------------------------
# LIVE UPDATE (AUTO-REFRESH)
# -----------------------------------
if st.session_state.is_detecting and st.session_state.detector:
    chord = st.session_state.detector.current_chord
    confidence = st.session_state.detector.confidence

    if confidence > 0.5:
        placeholder_chord.markdown(f"### 🎸 Current Chord: **{chord}**")
    else:
        placeholder_chord.markdown("### 🎸 Current Chord: **Listening...**")
    
    placeholder_conf.markdown(f"**Confidence:** {confidence:.2f}")
    try:
        placeholder_bar.progress(int(min(max(confidence, 0.0), 1.0) * 100))
    except Exception:
        placeholder_bar.progress(0)

    dh = st.session_state.detector.detection_history
    if dh:
        hist_list = list(dh)
        high_conf_history = [d for d in hist_list if d.get('confidence', 0) > 0.5]
        history = high_conf_history[-5:]
        if history:
            history_table = [
                {"Chord": d["chord"], "Confidence": f"{d['confidence']:.2f}"}
                for d in reversed(history)
            ]
            placeholder_history.table(history_table)
        else:
            placeholder_history.markdown("*No high-confidence detections yet*")

    time.sleep(0.5)
    st.rerun()
else:
    placeholder_chord.markdown("### 🎸 Current Chord: —")
    placeholder_conf.markdown("**Confidence:** 0.00")
    placeholder_bar.progress(0)
