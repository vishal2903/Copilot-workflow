#!/usr/bin/env python3
"""
LiveKit Import Diagnostic Tool
Run this script to diagnose LiveKit import issues
"""

import sys
import subprocess
from pathlib import Path

def check_virtual_env():
    """Check if we're in the correct virtual environment"""
    print("=== VIRTUAL ENVIRONMENT CHECK ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Check if we're in .venv
    if ".venv" in sys.executable:
        print("[OK] Using virtual environment (.venv)")
    else:
        print("[WARNING] Not using expected .venv virtual environment")
    
    return ".venv" in sys.executable

def check_livekit_packages():
    """Check installed LiveKit packages"""
    print("\n=== LIVEKIT PACKAGES CHECK ===")
    
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                              capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        livekit_packages = [line for line in lines if 'livekit' in line.lower()]
        
        if livekit_packages:
            print("[PACKAGES] Installed LiveKit packages:")
            for pkg in livekit_packages:
                print(f"  {pkg}")
        else:
            print("[ERROR] No LiveKit packages found")
            
        return len(livekit_packages) > 0
        
    except Exception as e:
        print(f"[ERROR] Error checking packages: {e}")
        return False

def test_individual_imports():
    """Test each import statement individually"""
    print("\n=== INDIVIDUAL IMPORT TESTS ===")
    
    imports_to_test = [
        ("livekit", "from livekit import agents"),
        ("agents submodules", "from livekit.agents import AgentSession, Agent, RoomInputOptions, function_tool, RunContext"),
        ("openai plugin", "from livekit.plugins import openai as lk_openai"),
        ("silero plugin", "from livekit.plugins import silero"),
        ("noise_cancellation plugin", "from livekit.plugins import noise_cancellation")
    ]
    
    results = {}
    
    for name, import_stmt in imports_to_test:
        try:
            exec(import_stmt)
            print(f"[OK] {name}: SUCCESS")
            results[name] = True
        except ImportError as e:
            print(f"[FAIL] {name}: FAILED - {e}")
            results[name] = False
        except Exception as e:
            print(f"[ERROR] {name}: ERROR - {e}")
            results[name] = False
    
    return results

def test_full_import():
    """Test the full import block as used in gdrive_analyzer.py"""
    print("\n=== FULL IMPORT BLOCK TEST ===")
    
    try:
        from livekit import agents
        from livekit.agents import AgentSession, Agent, RoomInputOptions, function_tool, RunContext
        from livekit.plugins import openai as lk_openai, noise_cancellation, silero
        
        print("[SUCCESS] All imports successful - LIVEKIT_AVAILABLE = True")
        return True
        
    except ImportError as e:
        print(f"[FAIL] Import failed - LIVEKIT_AVAILABLE = False")
        print(f"   Error: {e}")
        return False

def suggest_fixes(results):
    """Suggest fixes based on test results"""
    print("\n=== SUGGESTED FIXES ===")
    
    missing_packages = []
    
    if not results.get("openai plugin", False):
        missing_packages.append("livekit-plugins-openai")
    
    if not results.get("silero plugin", False):
        missing_packages.append("livekit-plugins-silero")
        
    if not results.get("noise_cancellation plugin", False):
        missing_packages.append("livekit-plugins-noise-cancellation")
    
    if missing_packages:
        print("[FIX] Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
    else:
        print("[OK] All required packages appear to be installed")

def main():
    """Main diagnostic function"""
    print("LiveKit Import Diagnostic Tool")
    print("=" * 50)
    
    # Run all checks
    venv_ok = check_virtual_env()
    packages_ok = check_livekit_packages()
    import_results = test_individual_imports()
    full_import_ok = test_full_import()
    
    # Suggest fixes
    suggest_fixes(import_results)
    
    # Summary
    print("\n=== SUMMARY ===")
    if full_import_ok:
        print("[SUCCESS] All LiveKit imports working correctly!")
        print("   Your gdrive_analyzer.py livekit-server command should work now.")
    else:
        print("[FAIL] LiveKit imports still failing")
        print("   Follow the suggested fixes above.")
    
    print(f"\n[TIP] To test your fix: python gdrive_analyzer.py livekit-server --help")

if __name__ == "__main__":
    main()