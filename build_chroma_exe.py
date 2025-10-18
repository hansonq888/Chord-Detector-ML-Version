#!/usr/bin/env python3
"""
Build script for Chord Detector Chroma executable
This script automates the process of creating a distributable .exe file for the chroma model
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def clean_build_dirs():
    """Clean previous build artifacts"""
    print("Cleaning previous build artifacts...")
    
    dirs_to_clean = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"   Removed {dir_name}/")
    
    # Clean .pyc files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyc'):
                os.remove(os.path.join(root, file))

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'librosa',
        'numpy',
        'scipy',
        'sklearn',
        'soundfile',
        'pyaudio',
        'tkinter'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            else:
                __import__(package)
            print(f"   [OK] {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   [MISSING] {package}")
    
    if missing_packages:
        print(f"\n[ERROR] Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages with: pip install " + " ".join(missing_packages))
        return False
    
    print("All dependencies found!")
    return True

def check_files():
    """Check if required files exist"""
    print("Checking required files...")
    
    required_files = [
        'live_chord_detector_chroma_gui.py',
        'chroma_model/live_chord_detector_chroma.py',
        'chroma_model/model_chroma.pkl',
        'ChordDetectorChroma.spec'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   [OK] {file_path}")
        else:
            missing_files.append(file_path)
            print(f"   [MISSING] {file_path}")
    
    if missing_files:
        print(f"\n[ERROR] Missing files: {', '.join(missing_files)}")
        return False
    
    print("All required files found!")
    return True

def build_executable():
    """Build the executable using PyInstaller"""
    print("Building executable...")
    
    try:
        # Run PyInstaller with the spec file
        cmd = [
            sys.executable, '-m', 'PyInstaller',
            '--clean',
            '--noconfirm',
            'ChordDetectorChroma.spec'
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("Build completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Build failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error during build: {e}")
        return False

def create_distribution_package():
    """Create a distribution package with all necessary files"""
    print("Creating distribution package...")
    
    package_name = "ChordDetectorChroma_Package"
    package_dir = f"dist/{package_name}"
    
    # Create package directory
    os.makedirs(package_dir, exist_ok=True)
    
    # Copy executable
    exe_source = "dist/ChordDetectorChroma.exe"
    exe_dest = f"{package_dir}/ChordDetectorChroma.exe"
    
    if os.path.exists(exe_source):
        shutil.copy2(exe_source, exe_dest)
        print(f"   Copied executable to {package_dir}/")
    else:
        print(f"[ERROR] Executable not found at {exe_source}")
        return False
    
    # Copy chroma model
    model_source = "chroma_model/model_chroma.pkl"
    model_dest = f"{package_dir}/chroma_model/model_chroma.pkl"
    
    if os.path.exists(model_source):
        os.makedirs(f"{package_dir}/chroma_model", exist_ok=True)
        shutil.copy2(model_source, model_dest)
        print(f"   Copied chroma model to {package_dir}/chroma_model/")
    else:
        print(f"[WARNING] Chroma model not found at {model_source}")
    
    # Copy sample files if they exist
    if os.path.exists("Sample Tests"):
        shutil.copytree("Sample Tests", f"{package_dir}/Sample Tests", dirs_exist_ok=True)
        print(f"   Copied Sample Tests to {package_dir}/")
    
    if os.path.exists("Chords"):
        shutil.copytree("Chords", f"{package_dir}/Chords", dirs_exist_ok=True)
        print(f"   Copied Chords to {package_dir}/")
    
    # Create README
    readme_content = """Chord Detector Chroma - Executable Package

This package contains the Chord Detector Chroma application.

FILES INCLUDED:
- ChordDetectorChroma.exe: Main application executable
- chroma_model/model_chroma.pkl: Trained chroma model
- Sample Tests/: Sample audio files for testing
- Chords/: Test chord files

USAGE:
1. Double-click ChordDetectorChroma.exe to run the application
2. Click "Start Detection" to begin live chord detection
3. Play chords on your instrument and watch the detection results

REQUIREMENTS:
- Windows 10 or later
- Microphone or audio input device
- No additional software installation required

TROUBLESHOOTING:
- If the application doesn't start, make sure you have a microphone connected
- The application requires audio input to function
- Make sure Windows allows microphone access for this application

For more information, visit the project repository.
"""
    
    with open(f"{package_dir}/README.txt", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"   Created README.txt in {package_dir}/")
    
    # Create installer script
    install_script = """@echo off
echo Chord Detector Chroma - Installation
echo ====================================
echo.
echo This will install Chord Detector Chroma to your system.
echo.
pause
echo.
echo Installation complete!
echo You can now run ChordDetectorChroma.exe
echo.
pause
"""
    
    with open(f"{package_dir}/install.bat", 'w') as f:
        f.write(install_script)
    
    print(f"   Created install.bat in {package_dir}/")
    
    # Create zip file
    try:
        import zipfile
        zip_path = f"dist/{package_name}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(package_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_path = os.path.relpath(file_path, package_dir)
                    zipf.write(file_path, arc_path)
        
        print(f"   Created {package_name}.zip")
        
    except Exception as e:
        print(f"   [WARNING] Could not create zip file: {e}")
    
    print(f"\nDistribution package created: {package_dir}/")
    return True

def main():
    """Main build process"""
    print("=" * 60)
    print("Chord Detector Chroma - Executable Builder")
    print("=" * 60)
    
    # Step 1: Clean previous builds
    clean_build_dirs()
    
    # Step 2: Check dependencies
    if not check_dependencies():
        print("\n[ERROR] Missing dependencies. Please install them and try again.")
        return False
    
    # Step 3: Check required files
    if not check_files():
        print("\n[ERROR] Missing required files. Please ensure all files are present.")
        return False
    
    # Step 4: Build executable
    if not build_executable():
        print("\n[ERROR] Build failed. Check the error messages above.")
        return False
    
    # Step 5: Create distribution package
    if not create_distribution_package():
        print("\n[ERROR] Failed to create distribution package.")
        return False
    
    print("\n" + "=" * 60)
    print("BUILD COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Executable: dist/ChordDetectorChroma.exe")
    print(f"Package: dist/ChordDetectorChroma_Package/")
    print(f"Zip file: dist/ChordDetectorChroma_Package.zip")
    print("\nYou can now distribute the ChordDetectorChroma_Package folder or zip file.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
