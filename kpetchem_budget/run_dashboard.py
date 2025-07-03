#!/usr/bin/env python3
"""
Launch script for K-PetChem Carbon Budget Toolkit Dashboard.

This script helps users launch the correct dashboard version and handles
common import issues.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required modules are available."""
    required_modules = [
        'streamlit', 'pandas', 'numpy', 'plotly', 'pyarrow', 'scipy'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("❌ Missing required modules:")
        for module in missing_modules:
            print(f"   - {module}")
        print("\n💡 Install missing modules with:")
        print(f"   pip install {' '.join(missing_modules)}")
        return False
    
    return True

def check_file_structure():
    """Check if required files exist in the current directory."""
    required_files = [
        'data_layer.py',
        'parameter_space.py',
        'pathway.py',
        'simulator.py',
        'datastore.py',
        'dashboard/app_upgraded.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n💡 Make sure you're running this script from the kpetchem_budget directory")
        print("   Expected structure:")
        print("""
   kpetchem_budget/
   ├── data_layer.py
   ├── parameter_space.py
   ├── pathway.py
   ├── simulator.py
   ├── datastore.py
   ├── dashboard/
   │   ├── app.py
   │   └── app_upgraded.py
   └── run_dashboard.py (this script)
        """)
        return False
    
    return True

def main():
    """Main launcher function."""
    print("🏭 K-PetChem Carbon Budget Toolkit Dashboard Launcher")
    print("=" * 55)
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Check requirements
    print("\n🔍 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    print("✅ All required modules are available")
    
    # Check file structure
    print("\n🔍 Checking file structure...")
    if not check_file_structure():
        sys.exit(1)
    print("✅ All required files are present")
    
    # Choose dashboard version
    print("\n🚀 Choose dashboard version:")
    print("1. Basic Dashboard (app.py)")
    print("2. Advanced Dashboard with Monte Carlo (app_upgraded.py) [RECOMMENDED]")
    
    choice = input("\nEnter choice (1 or 2, default=2): ").strip()
    
    if choice == "1":
        dashboard_file = "dashboard/app.py"
        print("\n🎯 Launching Basic Dashboard...")
    else:
        dashboard_file = "dashboard/app_upgraded.py"
        print("\n🎯 Launching Advanced Dashboard...")
    
    # Launch Streamlit
    try:
        print(f"📊 Starting Streamlit with {dashboard_file}")
        print("💡 The dashboard will open in your web browser")
        print("💡 Press Ctrl+C to stop the server")
        print("-" * 50)
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", dashboard_file
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error launching dashboard: {e}")
        print("💡 Try installing streamlit: pip install streamlit")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n❌ Streamlit not found. Install it with: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()