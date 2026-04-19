from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import keyboard

from multi_ghost.recorder import MultiGhostRecorder, RecorderConfig


def parse_region(region_raw: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not region_raw:
        return None
    parts = [int(x.strip()) for x in region_raw.split(",")]
    if len(parts) != 4:
        raise ValueError("region must be: left,top,right,bottom")
    return tuple(parts)  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Record multiple leaderboard ghost runs from screen capture.")
    parser.add_argument("--dataset-root", default="dataset", help="Root folder for datasets.")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS target.")
    parser.add_argument("--region", type=str, default=None, help="Capture region: left,top,right,bottom")
    parser.add_argument("--ghost-id", type=int, default=1, help="Initial ghost id.")
    args = parser.parse_args()

    config = RecorderConfig(
        dataset_root=Path(args.dataset_root),
        fps=args.fps,
        region=parse_region(args.region),
    )

    recorder = MultiGhostRecorder(config)
    recorder.set_ghost_id(args.ghost_id)
    recorder.start_system()

    running = True
    switch_requested = threading.Event()

    def on_start():
        recorder.start_recording()
        print(f"[F1] Recording started | ghost_id={recorder.current_ghost_id}")

    def on_stop():
        recorder.stop_recording()
        print("[F2] Recording stopped and run saved.")

    def on_save_index():
        index = recorder.save_dataset_index()
        print(f"[F3] Dataset index saved: {index}")

    def on_switch():
        switch_requested.set()

    def on_exit():
        nonlocal running
        running = False

    keyboard.add_hotkey("f1", on_start)
    keyboard.add_hotkey("f2", on_stop)
    keyboard.add_hotkey("f3", on_save_index)
    keyboard.add_hotkey("f4", on_switch)
    keyboard.add_hotkey("esc", on_exit)

    print("=" * 70)
    print("Multi-Ghost Recorder")
    print("F1=start recording | F2=stop recording | F3=save dataset index | F4=switch ghost_id | Esc=quit")
    print(f"Current ghost_id: {recorder.current_ghost_id}")
    print("=" * 70)

    frame_dt = 1.0 / max(1, args.fps)

    try:
        while running:
            tick_start = time.perf_counter()
            recorder.tick()

            if switch_requested.is_set():
                switch_requested.clear()
                if recorder.recording:
                    print("[F4] Stop current recording first (F2), then switch ghost_id.")
                else:
                    try:
                        new_id = int(input("Enter new ghost_id: ").strip())
                        recorder.set_ghost_id(new_id)
                        print(f"Switched to ghost_id={new_id}")
                    except Exception as exc:
                        print(f"Invalid ghost_id: {exc}")

            elapsed = time.perf_counter() - tick_start
            to_sleep = frame_dt - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
    finally:
        recorder.stop_recording()
        idx = recorder.save_dataset_index()
        recorder.stop_system()
        keyboard.unhook_all_hotkeys()
        print(f"Stopped recorder. Final dataset index: {idx}")


if __name__ == "__main__":
    main()
