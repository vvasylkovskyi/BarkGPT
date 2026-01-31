import datetime

class Logger:
    COLORS = {
        "info": "\033[94m",    # bright blue
        "success": "\033[92m", # bright green
        "warning": "\033[93m", # yellow
        "error": "\033[91m",   # bright red
        "debug": "\033[90m",   # bright black (gray)
        "reset": "\033[0m"     # reset
    }

    def __init__(self, name: str = None):
        self.name = name

    def _log(self, message: str, level: str):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color = self.COLORS.get(level, self.COLORS["info"])
        prefix = f"[{self.name}]" if self.name else ""
        reset = self.COLORS["reset"]
        print(f"{color}{now} [{level.upper()}]{prefix} {message}{reset}")

    def info(self, message: str):
        self._log(message, "info")

    def success(self, message: str):
        self._log(message, "success")

    def warning(self, message: str):
        self._log(message, "warning")

    def error(self, message: str):
        self._log(message, "error")

    def debug(self, message: str):
        self._log(message, "debug")
