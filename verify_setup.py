#!/usr/bin/env python3
"""
Setup Verification Script for YouTube Auto Pipeline
Checks all components and provides detailed feedback
"""

import os
import sys
import json
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Use Python 3.11+")
        return False

def check_required_files():
    """Check if all required files exist"""
    print("\n📁 Checking required files...")
    required_files = [
        (".env", "Environment variables file"),
        ("service_account.json", "Google Cloud service account key"),
        ("client_secret.json", "YouTube OAuth credentials"),
        ("token.pickle", "YouTube access token")
    ]
    
    all_exist = True
    for filename, description in required_files:
        if os.path.exists(filename):
            print(f"✅ {filename} - {description}")
        else:
            print(f"❌ {filename} - {description} (MISSING)")
            all_exist = False
    
    return all_exist

def check_environment_variables():
    """Check if environment variables are set"""
    print("\n🔧 Checking environment variables...")
    
    # Load .env file
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    # Read and check variables
    with open(env_file, 'r') as f:
        content = f.read()
    
    required_vars = [
        "OPENAI_API_KEY",
        "ELEVENLABS_API_KEY", 
        "HINDI_VOICE_ID"
    ]
    
    all_set = True
    for var in required_vars:
        if f"{var}=" in content and not f"{var}=" in content.split('\n')[0]:
            print(f"✅ {var} - Set")
        else:
            print(f"❌ {var} - Not set or empty")
            all_set = False
    
    return all_set

def check_dependencies():
    """Check if all Python dependencies are installed"""
    print("\n📦 Checking Python dependencies...")
    
    required_packages = [
        "openai",
        "elevenlabs", 
        "gspread",
        "google",
        "moviepy",
        "pydub",
        "pillow",
        "requests",
        "python-dotenv"
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - Installed")
        except ImportError:
            print(f"❌ {package} - Not installed")
            all_installed = False
    
    return all_installed

def check_comfyui_connection():
    """Check if ComfyUI is running and accessible"""
    print("\n🎨 Checking ComfyUI connection...")
    
    try:
        import requests
        response = requests.get("http://127.0.0.1:8188", timeout=5)
        if response.status_code == 200:
            print("✅ ComfyUI is running on http://127.0.0.1:8188")
            return True
        else:
            print(f"⚠️ ComfyUI responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("❌ ComfyUI is not running or not accessible")
        print("   Start ComfyUI with: python main.py --listen 0.0.0.0 --port 8188")
        return False

def check_workflow_files():
    """Check if workflow files exist"""
    print("\n⚙️ Checking workflow files...")
    
    workflow_dir = Path("workflow")
    if not workflow_dir.exists():
        print("❌ workflow/ directory not found")
        return False
    
    required_workflows = [
        "prompt.json",
        "prompt_advanced.json", 
        "prompt_ultra_quality.json"
    ]
    
    all_exist = True
    for workflow in required_workflows:
        workflow_path = workflow_dir / workflow
        if workflow_path.exists():
            print(f"✅ {workflow} - Found")
        else:
            print(f"❌ {workflow} - Missing")
            all_exist = False
    
    return all_exist

def check_background_music():
    """Check if background music files exist"""
    print("\n🎵 Checking background music...")
    
    bg_dir = Path("backgrounds")
    if not bg_dir.exists():
        print("❌ backgrounds/ directory not found")
        return False
    
    music_files = list(bg_dir.glob("*.mp3"))
    if music_files:
        print(f"✅ {len(music_files)} background music files found")
        for file in music_files[:3]:  # Show first 3
            print(f"   - {file.name}")
        if len(music_files) > 3:
            print(f"   ... and {len(music_files) - 3} more")
        return True
    else:
        print("❌ No background music files found")
        return False

def check_google_sheets_access():
    """Check if Google Sheets API is accessible"""
    print("\n📊 Checking Google Sheets access...")
    
    try:
        from main import sheet_ws
        ws = sheet_ws()
        print("✅ Google Sheets API is accessible")
        return True
    except Exception as e:
        print(f"❌ Google Sheets API error: {e}")
        return False

def check_youtube_api():
    """Check if YouTube API is accessible"""
    print("\n🎬 Checking YouTube API...")
    
    try:
        import pickle
        with open("token.pickle", "rb") as f:
            creds = pickle.load(f)
        print("✅ YouTube API credentials loaded")
        return True
    except Exception as e:
        print(f"❌ YouTube API error: {e}")
        return False

def main():
    """Run all verification checks"""
    print("🔍 YouTube Auto Pipeline - Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Files", check_required_files),
        ("Environment Variables", check_environment_variables),
        ("Dependencies", check_dependencies),
        ("ComfyUI Connection", check_comfyui_connection),
        ("Workflow Files", check_workflow_files),
        ("Background Music", check_background_music),
        ("Google Sheets API", check_google_sheets_access),
        ("YouTube API", check_youtube_api)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
    
    print(f"\n🎯 Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Your setup is ready.")
        print("🚀 You can now run: python main.py")
    else:
        print("⚠️ Some checks failed. Please review the issues above.")
        print("📖 See docs/SETUP.md for detailed setup instructions.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 