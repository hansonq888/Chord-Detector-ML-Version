#!/usr/bin/env python3
"""
Installation script for Live Chord Detector
This script will install all required dependencies for live chord detection.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def check_package(package):
    """Check if a package is already installed"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def main():
    """Main installation function"""
    print("🎵 Live Chord Detector - Installation Script")
    print("=" * 50)
    
    # Required packages
    packages = [
        ("librosa", "librosa>=0.10.0"),
        ("numpy", "numpy>=1.21.0"),
        ("sklearn", "scikit-learn>=1.0.0"),
        ("joblib", "joblib>=1.1.0"),
        ("pyaudio", "pyaudio>=0.2.11")
    ]
    
    print("Checking and installing required packages...")
    print()
    
    failed_packages = []
    
    for import_name, pip_name in packages:
        if check_package(import_name):
            print(f"✅ {import_name} is already installed")
        else:
            print(f"📦 Installing {pip_name}...")
            if not install_package(pip_name):
                failed_packages.append(pip_name)
    
    print()
    
    if failed_packages:
        print("❌ Some packages failed to install:")
        for package in failed_packages:
            print(f"   - {package}")
        print()
        print("You may need to install these manually:")
        print("pip install " + " ".join(failed_packages))
        print()
        print("For PyAudio on Windows, you might need:")
        print("pip install pipwin")
        print("pipwin install pyaudio")
    else:
        print("🎉 All packages installed successfully!")
        print()
        print("You can now run the live chord detector:")
        print("  python live_chord_detector.py          # Command line version")
        print("  python live_chord_detector_gui.py      # GUI version")
        print()
        print("Make sure you have a trained model (model.pkl) in the directory.")
        print("If not, run: python train_model.py")

if __name__ == "__main__":
    main()
