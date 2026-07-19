#!/usr/bin/env python3
"""
🐟 Pelagic Fish Age Predictor - One-Click Launcher
Windows-friendly version with proper error handling
"""

import subprocess
import sys
import os
import importlib.util
from pathlib import Path
import time

# ANSI color codes for better terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

# Windows doesn't support ANSI colors in cmd by default
if sys.platform == 'win32':
    try:
        import colorama
        colorama.init()
    except ImportError:
        # If colorama not installed, disable colors
        GREEN = YELLOW = RED = BLUE = RESET = BOLD = ''

def print_banner():
    """Print the application banner"""
    banner = f"""
{BLUE}{BOLD}
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   🐟  PELAGIC FISH AGE PREDICTOR                        ║
║                                                          ║
║   Machine Learning Web Application                      ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
{RESET}
"""
    print(banner)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"{RED}❌ Python 3.8+ is required. You have {version.major}.{version.minor}{RESET}")
        return False
    print(f"{GREEN}✅ Python {version.major}.{version.minor}.{version.micro}{RESET}")
    return True

def check_package(package_name, import_name=None):
    """Check if a Python package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            return True
        return False
    except ImportError:
        return False

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'sklearn',
        'matplotlib': 'matplotlib',
        'PIL': 'PIL',
        'joblib': 'joblib',
        'seaborn': 'seaborn'
    }
    
    missing = []
    for package, import_name in required_packages.items():
        if not check_package(package, import_name):
            missing.append(package)
    
    if missing:
        print(f"{YELLOW}⚠️ Missing packages: {', '.join(missing)}{RESET}")
        print(f"\n{BOLD}To install: {RESET}pip install -r requirements.txt")
        return False
    
    print(f"{GREEN}✅ All dependencies installed{RESET}")
    return True

def check_model_file():
    """Check if the model file exists"""
    model_paths = [
        Path("scripts/fish_age_model.pkl"),
        Path("fish_age_model.pkl")
    ]
    
    for path in model_paths:
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"{GREEN}✅ Model file found: {path} ({size:.2f} MB){RESET}")
            return True
    
    print(f"{RED}❌ Model file 'fish_age_model.pkl' not found{RESET}")
    print(f"{YELLOW}Expected locations: scripts/fish_age_model.pkl{RESET}")
    return False

def check_data_files():
    """Check if data files exist"""
    data_files = [
        "data/pelagic_training_set.csv",
        "data/pelagic_test_set.csv"
    ]
    
    missing = []
    for file_path in data_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if missing:
        print(f"{YELLOW}⚠️ Missing data files: {', '.join(missing)}{RESET}")
        print(f"{YELLOW}The app will still run but some features may be limited{RESET}")
        return False
    
    print(f"{GREEN}✅ Data files found{RESET}")
    return True

def check_output_files():
    """Check if output visualization files exist"""
    output_dir = Path("output")
    if not output_dir.exists():
        print(f"{YELLOW}⚠️ Output directory not found{RESET}")
        return False
    
    png_files = list(output_dir.glob("*.png"))
    if png_files:
        print(f"{GREEN}✅ Found {len(png_files)} visualization images{RESET}")
    else:
        print(f"{YELLOW}⚠️ No visualization images found in output/{RESET}")
    
    return True

def run_streamlit():
    """Launch the Streamlit application"""
    print(f"\n{BOLD}🚀 Launching application...{RESET}\n")
    
    # Check if streamlit is available
    try:
        import streamlit
    except ImportError:
        print(f"{RED}❌ Streamlit not installed{RESET}")
        print(f"{BOLD}To install: {RESET}pip install streamlit")
        return False
    
    # Determine the app path
    app_paths = [
        Path("scripts/front_end.py"),
        Path("front_end.py")
    ]
    
    app_path = None
    for path in app_paths:
        if path.exists():
            app_path = path
            break
    
    if app_path is None:
        print(f"{RED}❌ front_end.py not found{RESET}")
        print(f"{YELLOW}Expected location: scripts/front_end.py{RESET}")
        return False
    
    print(f"{GREEN}✅ Found app: {app_path}{RESET}\n")
    print(f"{BLUE}{BOLD}Press Ctrl+C to stop the server{RESET}\n")
    print(f"{YELLOW}Opening in browser...{RESET}\n")
    
    # Launch Streamlit
    try:
        # For Windows, use python -m streamlit
        if sys.platform == 'win32':
            cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
        else:
            cmd = ["streamlit", "run", str(app_path)]
        
        # Add common options
        cmd.extend(["--server.port", "8501"])
        cmd.extend(["--server.address", "localhost"])
        
        # Use subprocess with proper handling
        process = subprocess.Popen(cmd)
        process.wait()
        return True
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}🛑 Application stopped{RESET}")
        return True
    except Exception as e:
        print(f"{RED}❌ Error launching app: {e}{RESET}")
        return False

def main():
    """Main execution function"""
    print_banner()
    
    print(f"{BOLD}🔍 Running pre-flight checks...{RESET}\n")
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version()),
        ("Dependencies", check_requirements()),
        ("Model File", check_model_file()),
        ("Data Files", check_data_files()),
        ("Output Files", check_output_files())
    ]
    
    # Check if any critical checks failed
    critical_failures = False
    for check_name, passed in checks:
        if not passed and check_name in ["Python Version", "Dependencies", "Model File"]:
            critical_failures = True
    
    print()
    
    if critical_failures:
        print(f"{RED}❌ Critical checks failed. Please fix the issues above.{RESET}")
        print(f"{YELLOW}Try running: pip install -r requirements.txt{RESET}")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Launch the app
    success = run_streamlit()
    
    if not success:
        print(f"{RED}❌ Failed to launch application{RESET}")
        input("\nPress Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{RED}❌ Unexpected error: {e}{RESET}")
        input("\nPress Enter to exit...")
        sys.exit(1)