import sys
import os
from datetime import datetime
 
 
LOGS_DIR = "results/logs/test_run_log"
 
 
class _Tee:
    """Mirrors writes to both a file and the original stdout."""
 
    def __init__(self, log_path: str, original_stdout):
        self.file     = open(log_path, "a", encoding="utf-8")
        self.terminal = original_stdout
 
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()
 
    def flush(self):
        self.terminal.flush()
        self.file.flush()
 
    def close(self):
        self.file.close()
 
 
_tee_instance = None
 
 
def setup_logger(name: str):
    """
    Redirect all print() output to both terminal and results/logs/<name>.log.
    Call once at the start of each script.
 
    Args:
        name: Log file stem, e.g. "extraction" → results/logs/extraction.log
    """
    global _tee_instance
 
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f"{name}.log")
 
    _tee_instance = _Tee(log_path, sys.stdout)
    sys.stdout = _tee_instance
 
    print(f"\n")
    print(f"  Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Log: {log_path}")
    print(f"\n")
 
 
def close_logger():
    """Restore original stdout and close the log file."""
    global _tee_instance
    if _tee_instance is not None:
        sys.stdout = _tee_instance.terminal
        _tee_instance.close()
        _tee_instance = None