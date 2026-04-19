from __future__ import annotations

import csv
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import keyboard

try:
    import dxcam  # type: ignore
except Exception:  # pragma: no cover
    dxcam = None

try:
    import mss  # type: ignore
except Exception:  # pragma: no cover
    mss = None


ACTION_KEYS = ["w", "a", "s", "d", "q", "e", "space", "shift", "ctrl"]
ACTION_COLUMNS = ["w", "a", "s", "d", "q", "e", "space", "shift_ctrl"]


@dataclass
class RecorderConfig:
    dataset_root: Path = Path("dataset")
    fps: int = 30
    region: Optional[Tuple[int, int, int, int]] = None  # left, top, right, bottom
    jpg_quality: int = 95


class ScreenCapture:
    """dxcam-first capture with an mss fallback."""

    def __init__(self, fps: int, region: Optional[Tuple[int, int, int, int]] = None) -> None:
        self.fps = fps
        self.region = region
        self._camera = None
        self._mss_ctx = None
        self._monitor = None
        self._mode = None

    def start(self) -> None:
        if dxcam is not None:
            self._camera = dxcam.create(output_color="BGR")
            self._camera.start(target_fps=self.fps, video_mode=True, region=self.region)
            self._mode = "dxcam"
            return

        if mss is None:
            raise RuntimeError("Neither dxcam nor mss is available. Install dxcam for Windows capture.")

        self._mss_ctx = mss.mss()
        mon = self._mss_ctx.monitors[1]
        if self.region is None:
            self._monitor = {
                "left": mon["left"],
                "top": mon["top"],
                "width": mon["width"],
                "height": mon["height"],
            }
        else:
            l, t, r, b = self.region
            self._monitor = {"left": l, "top": t, "width": r - l, "height": b - t}
        self._mode = "mss"

    def get_latest_frame(self):
        if self._mode == "dxcam" and self._camera is not None:
            return self._camera.get_latest_frame()
        if self._mode == "mss" and self._mss_ctx is not None and self._monitor is not None:
            raw = self._mss_ctx.grab(self._monitor)
            frame = cv2.cvtColor(cv2.UMat(raw.rgb), cv2.COLOR_RGB2BGR).get()
            return frame
        return None

    def stop(self) -> None:
        if self._camera is not None:
            self._camera.stop()
            self._camera = None
        if self._mss_ctx is not None:
            self._mss_ctx.close()
            self._mss_ctx = None


class AsyncFrameWriter:
    def __init__(self, jpg_quality: int = 95, max_queue: int = 1024) -> None:
        self.jpg_quality = jpg_quality
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._stop_event = threading.Event()
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._thread.start()
            self._started = True

    def enqueue(self, frame, frame_path: Path) -> None:
        self._queue.put((frame, frame_path))

    def flush(self) -> None:
        self._queue.join()

    def stop(self) -> None:
        self.flush()
        self._stop_event.set()
        self._queue.put((None, None))
        self._thread.join(timeout=5)

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            frame, frame_path = self._queue.get()
            try:
                if frame is None:
                    return
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality])
            finally:
                self._queue.task_done()


class MultiGhostRecorder:
    def __init__(self, config: RecorderConfig) -> None:
        self.config = config
        self.dataset_root = config.dataset_root
        self.dataset_root.mkdir(parents=True, exist_ok=True)

        self.capture = ScreenCapture(config.fps, config.region)
        self.writer = AsyncFrameWriter(jpg_quality=config.jpg_quality)

        self.current_ghost_id = 1
        self.recording = False
        self.run_id: Optional[str] = None
        self.run_dir: Optional[Path] = None
        self.frames_dir: Optional[Path] = None

        self.records: List[Dict] = []
        self.frame_idx = 0
        self.start_time = 0.0

    def start_system(self) -> None:
        self.capture.start()
        self.writer.start()

    def stop_system(self) -> None:
        self.writer.stop()
        self.capture.stop()

    def set_ghost_id(self, ghost_id: int) -> None:
        if self.recording:
            raise RuntimeError("Cannot switch ghost while recording. Press F2 first.")
        self.current_ghost_id = ghost_id

    def start_recording(self) -> None:
        if self.recording:
            return
        ghost_dir = self.dataset_root / f"ghost_{self.current_ghost_id}"
        runs_dir = ghost_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        run_number = self._next_run_number(runs_dir)
        self.run_id = f"run_{run_number:04d}"
        self.run_dir = runs_dir / self.run_id
        self.frames_dir = self.run_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.records = []
        self.frame_idx = 0
        self.start_time = time.time()
        self.recording = True

    def stop_recording(self) -> None:
        if not self.recording:
            return
        self._save_current_run()
        self.recording = False
        self.run_id = None
        self.run_dir = None
        self.frames_dir = None

    def save_dataset_index(self) -> Path:
        index_path = self.dataset_root / "index.csv"
        rows: List[Dict] = []
        for csv_path in sorted(self.dataset_root.glob("ghost_*/runs/run_*/actions.csv")):
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows.extend(reader)

        if rows:
            fieldnames = list(rows[0].keys())
            with index_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        return index_path

    def tick(self) -> None:
        if not self.recording:
            return
        frame = self.capture.get_latest_frame()
        if frame is None or self.frames_dir is None or self.run_id is None:
            return

        t = time.time()
        rel_ts = t - self.start_time
        frame_name = f"{self.frame_idx:06d}.jpg"
        frame_path = self.frames_dir / frame_name

        self.writer.enqueue(frame, frame_path)

        actions = self._read_actions()
        self.records.append(
            {
                "timestamp": f"{rel_ts:.6f}",
                "ghost_id": self.current_ghost_id,
                "run_id": self.run_id,
                "frame_idx": self.frame_idx,
                "frame_path": str(frame_path.relative_to(self.dataset_root)).replace("\\", "/"),
                **actions,
            }
        )
        self.frame_idx += 1

    def _save_current_run(self) -> None:
        if self.run_dir is None:
            return
        self.writer.flush()
        actions_csv = self.run_dir / "actions.csv"
        fieldnames = ["timestamp", "ghost_id", "run_id", "frame_idx", "frame_path", *ACTION_COLUMNS]
        with actions_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

    @staticmethod
    def _next_run_number(runs_dir: Path) -> int:
        ids = []
        for p in runs_dir.glob("run_*"):
            try:
                ids.append(int(p.name.split("_")[-1]))
            except ValueError:
                continue
        return max(ids, default=0) + 1

    @staticmethod
    def _read_actions() -> Dict[str, float]:
        shift_val = 1.0 if keyboard.is_pressed("shift") else 0.0
        ctrl_val = 1.0 if keyboard.is_pressed("ctrl") else 0.0
        return {
            "w": float(keyboard.is_pressed("w")),
            "a": float(keyboard.is_pressed("a")),
            "s": float(keyboard.is_pressed("s")),
            "d": float(keyboard.is_pressed("d")),
            "q": float(keyboard.is_pressed("q")),
            "e": float(keyboard.is_pressed("e")),
            "space": float(keyboard.is_pressed("space")),
            "shift_ctrl": max(shift_val, ctrl_val),
        }
