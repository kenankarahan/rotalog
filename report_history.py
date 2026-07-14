import os
import sys
import json

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HISTORY_FILE_PATH = os.path.join(BASE_DIR, "reports_history.json")

def load_report_history():
    if os.path.exists(HISTORY_FILE_PATH):
        with open(HISTORY_FILE_PATH, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def save_report_history(history_list):
    with open(HISTORY_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(history_list, f, ensure_ascii=False, indent=4)

def clear_report_history():
    if os.path.exists(HISTORY_FILE_PATH):
        os.remove(HISTORY_FILE_PATH)

def delete_report_by_index(index):
    history_list = load_report_history()
    if 0 <= index < len(history_list):
        history_list.pop(index)
        save_report_history(history_list)