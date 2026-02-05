import time


def progress_bar(i, total, start_time, prefix=""):
    frac = (i + 1) / total
    bar_len = 40
    filled = int(bar_len * frac)
    bar = "█" * filled + "-" * (bar_len - filled)

    elapsed = time.time() - start_time
    rate = (i + 1) / elapsed if elapsed > 0 else 0
    eta = (total - (i + 1)) / rate if rate > 0 else 0

    print(
        f"\r{prefix}: {frac*100:6.2f}%|{bar}| "
        f"{i+1}/{total} "
        f"[{elapsed:5.1f}s<{eta:5.1f}s, {rate:7.2f} it/s]",
        end="",
        flush=True,
    )

    if i + 1 == total:
        print()
