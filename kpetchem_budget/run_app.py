#!/usr/bin/env python3
"""
Launcher script for Korean Petrochemical Carbon Budget Streamlit app.

Usage:
    python run_app.py
    
Or from command line:
    streamlit run app.py
"""

import sys
import os
import subprocess

def main():
    """Launch the Streamlit app."""
    # Add current directory to Python path
    current_dir = os.path.dirname(__file__)
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    try:
        # Try to import required modules
        import streamlit
        from app import main as app_main
        
        print("Starting Korean Petrochemical Carbon Budget App...")
        print("Navigate to the URL shown below in your browser.")
        print("-" * 50)
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            os.path.join(current_dir, "app.py")
        ])
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install streamlit pandas numpy matplotlib scipy")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()