import sys
import subprocess
import importlib.util
from pathlib import Path
import json

BASE = Path(__file__).resolve().parent

# Paths to your modules
TEXT_PIPELINE_MODULE_PATH = BASE / "text_pipeline" / "analyzer.py"
FACE_IMG_MODULE_PATH = BASE / "face_emotion.py"
WEBCAM_MODULE_PATH = BASE / "realtimedetection.py"
ALT_WEBCAM_PATH = BASE / "emotion_detector.py"

PY = sys.executable

def import_from_path(path: Path):
    """Import module from file path and return module object (executes top-level code)."""
    if not path.exists():
        return None
    name = path.stem
    # If already imported, return it
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_as_subprocess(path: Path, args=None):
    args = args or []
    cmd = [PY, str(path)] + args
    subprocess.run(cmd)

def try_call_or_subprocess(path: Path, func_names):
    """
    Try to import module and call the first callable found in func_names.
    If none found, run file as subprocess.
    """
    mod = import_from_path(path)
    if mod:
        for fname in func_names:
            if hasattr(mod, fname):
                attr = getattr(mod, fname)
                if callable(attr):
                    try:
                        return attr()
                    except TypeError:
                        return attr()
    return run_as_subprocess(path)

def text_analysis_flow():
    """Interactive text analysis fallback: call analyze_text if present, otherwise run file."""
    mod = import_from_path(TEXT_PIPELINE_MODULE_PATH)
    if mod and hasattr(mod, "analyze_text"):
        analyze = getattr(mod, "analyze_text")
        print("Entering interactive text analyzer. Type 'exit' to go back.")
        while True:
            txt = input("\nEnter text (or 'exit'): ").strip()
            if txt.lower() == "exit":
                break
            try:
                res = analyze(txt)
                # Pretty print if it's dict-like
                if isinstance(res, (dict, list)):
                    print(json.dumps(res, indent=2, ensure_ascii=False))
                else:
                    print(res)
            except Exception as e:
                print("Error calling analyze_text():", e)
    else:
        print("analyze_text() not found in text_pipeline/analyzer.py — running file as script.")
        run_as_subprocess(TEXT_PIPELINE_MODULE_PATH)

def image_face_flow():
    """Interactive image face/emotion flow."""
    mod = import_from_path(FACE_IMG_MODULE_PATH)
    if mod:
        for fname in ("analyze_face", "predict", "main", "run"):
            if hasattr(mod, fname):
                try:
                    return getattr(mod, fname)()
                except Exception as e:
                    print(f"Error calling {fname}(): {e}")
                    break
    print("Launching face_emotion.py as script.")
    run_as_subprocess(FACE_IMG_MODULE_PATH)

def webcam_flow():
    """Start webcam realtime detection. Prefer realtimedetection.py, else emotion_detector.py"""
    chosen = WEBCAM_MODULE_PATH if WEBCAM_MODULE_PATH.exists() else ALT_WEBCAM_PATH
    if not chosen.exists():
        print("No webcam module found (realtimedetection.py or emotion_detector.py).")
        return
    mod = import_from_path(chosen)
    if mod:
        for fname in ("main", "run", "start"):
            if hasattr(mod, fname):
                try:
                    return getattr(mod, fname)()
                except Exception as e:
                    print(f"Error calling {fname}(): {e}")
                    break
    print(f"Running {chosen.name} as subprocess (it may open the webcam window).")
    run_as_subprocess(chosen)

def show_menu():
    print("\nEMOSENAI — Integrated Launcher")
    print("1) Text analyzer")
    print("2) Image face/emotion analyzer")
    print("3) Webcam realtime detection")
    print("4) Exit")

def main():
    if not (BASE / "text_pipeline").exists():
        print("Warning: text_pipeline/ directory not found.")
    while True:
        show_menu()
        choice = input("\nChoose (1-4): ").strip()
        if choice == "1":
            text_analysis_flow()
        elif choice == "2":
            image_face_flow()
        elif choice == "3":
            webcam_flow()
        elif choice == "4" or choice.lower() in ("q", "quit", "exit"):
            print("Exiting.")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
