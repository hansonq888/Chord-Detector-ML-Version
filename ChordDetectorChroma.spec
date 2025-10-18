# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files

# Collect data files for librosa and other dependencies
datas = []

# Add chroma model file if it exists
if os.path.exists('chroma_model/model_chroma.pkl'):
    datas.append(('chroma_model/model_chroma.pkl', 'chroma_model'))

# Add sample test files
if os.path.exists('Sample Tests'):
    datas.append(('Sample Tests', 'Sample Tests'))

# Add Chords folder for testing
if os.path.exists('Chords'):
    datas.append(('Chords', 'Chords'))

# Collect librosa data files
try:
    librosa_datas = collect_data_files('librosa')
    datas.extend(librosa_datas)
except:
    pass

# Collect scikit-learn data files
try:
    sklearn_datas = collect_data_files('sklearn')
    datas.extend(sklearn_datas)
except:
    pass

a = Analysis(
    ['live_chord_detector_chroma_gui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'librosa',
        'librosa.feature',
        'librosa.effects',
        'librosa.util',
        'sklearn',
        'sklearn.ensemble',
        'sklearn.model_selection',
        'sklearn.preprocessing',
        'numpy',
        'scipy',
        'soundfile',
        'audioread',
        'pyaudio',
        'tkinter',
        'tkinter.ttk',
        'threading',
        'collections',
        'time',
        'chroma_model.live_chord_detector_chroma'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'pandas',
        'jupyter',
        'IPython',
        'notebook'
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ChordDetectorChroma',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
