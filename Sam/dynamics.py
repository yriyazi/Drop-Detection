"""
    Author:         Yassin Riyazi
    Date:           10.01.2026

    Description: 
        Interactive SAM 2 video tracking driver.

        The script flattens batched frame folders, lets the user annotate an initial
        frame with point and box prompts, and then propagates those prompts through the
        full sequence using the SAM 2 video predictor. During propagation it can export
        PNG overlays, compressed mask tensors (``masks_obj_XXX.npz`` written next to
        each ``sam2_tracking`` folder), and periodic supervision frames whose stride is
        controlled via ``--supervision-stride`` (default: 20 frames).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    from sam2.build_sam import build_sam2_video_predictor
except:
    raise ImportError("Failed to import SAM 2 modules. Ensure that SAM 2 is installed")


@dataclass
class Prompt:
    points: Optional[np.ndarray]
    labels: Optional[np.ndarray]
    box: Optional[np.ndarray]


@dataclass
class TrackingResult:
    centroids: Dict[int, List[Tuple[int, float, float]]]
    last_masks: Dict[int, np.ndarray]


def load_prompts_json(path: str) -> Dict[int, Prompt]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    prompts: Dict[int, Prompt] = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid prompts JSON (expected object): {path}")
    for key, entry in raw.items():
        try:
            obj_id = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid object id '{key}' in {path}")
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid entry for object {obj_id} in {path}")

        points = entry.get("points")
        labels = entry.get("labels")
        box = entry.get("box")

        points_arr = (
            np.array(points, dtype=np.float32)
            if points is not None and len(points) > 0
            else None
        )
        labels_arr = (
            np.array(labels, dtype=np.int32)
            if labels is not None and len(labels) > 0
            else None
        )
        box_arr = np.array(box, dtype=np.float32) if box is not None else None
        prompts[obj_id] = Prompt(points=points_arr, labels=labels_arr, box=box_arr)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def find_roi_prompts_file(
    frame_root: str,
    output_dir: Optional[str],
    roi_file: Optional[str],
    roi_name: str,
) -> Optional[str]:
    candidates: List[str] = []
    if roi_file:
        candidates.append(roi_file)
    candidates.append(os.path.join(os.path.abspath(frame_root), roi_name))
    if output_dir:
        candidates.append(os.path.join(os.path.abspath(output_dir), roi_name))
        candidates.append(os.path.join(os.path.abspath(output_dir), "prompts.json"))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


_FRAME_RE = re.compile(r"(\d+)")
_SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def extract_frame_number(path: str) -> int:
    name = os.path.splitext(os.path.basename(path))[0]
    matches = _FRAME_RE.findall(name)
    if matches:
        return int(matches[-1])
    return sys.maxsize


def collect_frame_paths(
    root: str,
    batch_pattern: str,
    max_frames: Optional[int] = None,
) -> List[str]:
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Frame root not found: {root}")

    direct_frames: List[str] = []
    for ext in _SUPPORTED_EXTS:
        direct_frames.extend(glob_no_raise(os.path.join(root, f"*{ext}")))
    if direct_frames:
        frame_paths = sorted(direct_frames, key=lambda p: (extract_frame_number(p), p))
    else:
        batch_dirs = sorted(
            glob_no_raise(os.path.join(root, batch_pattern)),
            key=lambda p: (extract_frame_number(p), p),
        )
        if not batch_dirs:
            raise RuntimeError(
                "Frame root does not contain image files or batch directories."
            )
        frame_paths = []
        for batch_dir in batch_dirs:
            batch_frames: List[str] = []
            for ext in _SUPPORTED_EXTS:
                batch_frames.extend(glob_no_raise(os.path.join(batch_dir, f"*{ext}")))
            batch_frames = sorted(batch_frames, key=lambda p: (extract_frame_number(p), p))
            frame_paths.extend(batch_frames)
    if not frame_paths:
        raise RuntimeError(f"No frames found under {root}")
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]
    return frame_paths


def _sorted_batch_dirs(parent: str, batch_pattern: str) -> List[str]:
    candidates = [
        path
        for path in glob_no_raise(os.path.join(parent, batch_pattern))
        if os.path.isdir(path)
    ]
    return sorted(candidates, key=lambda p: (extract_frame_number(p), p))


def discover_contiguous_batches(frame_root: str, batch_pattern: str) -> List[str]:
    abs_root = os.path.abspath(frame_root)
    if not os.path.isdir(abs_root):
        raise FileNotFoundError(f"Batch root not found: {abs_root}")

    # Case 1: the provided directory already houses batch_* subdirectories.
    subdirs = _sorted_batch_dirs(abs_root, batch_pattern)
    if subdirs:
        return subdirs

    # Case 2: the provided directory is itself one of the batch_* directories.
    parent = os.path.dirname(abs_root)
    peer_dirs = _sorted_batch_dirs(parent, batch_pattern)
    if abs_root in peer_dirs:
        start_idx = peer_dirs.index(abs_root)
        return peer_dirs[start_idx:]

    # Case 3: fall back to treating the provided directory as a single batch.
    return [abs_root]


def derive_prompts_from_masks(
    masks: Dict[int, np.ndarray],
    tolerance_px: int,
) -> Dict[int, Prompt]:
    derived: Dict[int, Prompt] = {}
    for obj_id, mask in masks.items():
        if mask is None:
            continue
        mask_bool = np.asarray(mask, dtype=bool)
        if not mask_bool.any():
            continue
        ys, xs = np.where(mask_bool)
        min_x = int(xs.min())
        max_x = int(xs.max())
        min_y = int(ys.min())
        max_y = int(ys.max())
        height, width = mask_bool.shape
        padded = np.array(
            [
                max(0, min_x - tolerance_px),
                max(0, min_y - tolerance_px),
                min(width - 1, max_x + tolerance_px),
                min(height - 1, max_y + tolerance_px),
            ],
            dtype=np.float32,
        )
        centroid_x = float(xs.mean())
        centroid_y = float(ys.mean())
        derived[obj_id] = Prompt(
            points=np.array([[centroid_x, centroid_y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
            box=padded,
        )
    return derived


def glob_no_raise(pattern: str) -> List[str]:
    from glob import glob

    try:
        return glob(pattern)
    except re.error:
        return []


def flatten_frames(
    frame_paths: Sequence[str],
    dest_dir: Optional[str] = None,
    keep_dest: bool = False,
    allow_symlink: bool = True,
) -> Tuple[str, Callable[[], None]]:
    if dest_dir is None:
        dest_dir = tempfile.mkdtemp(prefix="sam2_frames_")
        cleanup = lambda: shutil.rmtree(dest_dir, ignore_errors=True)
    else:
        dest_dir = os.path.abspath(dest_dir)
        os.makedirs(dest_dir, exist_ok=True)
        if os.listdir(dest_dir):
            raise RuntimeError(f"Destination directory is not empty: {dest_dir}")
        cleanup = lambda: None
    os.makedirs(dest_dir, exist_ok=True)

    for idx, src in enumerate(frame_paths):
        dst_name = os.path.join(dest_dir, f"{idx:06d}.jpg")
        if os.path.exists(dst_name):
            continue
        abs_src = os.path.abspath(src)
        if allow_symlink:
            try:
                os.symlink(abs_src, dst_name)
                continue
            except OSError:
                pass
        shutil.copy2(abs_src, dst_name)

    if keep_dest:
        cleanup = lambda: None
    return dest_dir, cleanup


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_type = device_arg
    device = torch.device(device_type)
    return device


def configure_torch(device: torch.device) -> None:
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def configure_matplotlib() -> str:
    preferred = [
        "module://ipympl.backend_nbagg",
        "Qt5Agg",
        "TkAgg",
    ]
    for backend in preferred:
        try:
            matplotlib.use(backend)
            return backend
        except Exception:
            continue
    return matplotlib.get_backend()


class SAM2Selector:
    def __init__(self, ax: plt.Axes, max_objects: Optional[int] = None):
        self.ax = ax
        self.max_objects = max_objects
        self.objects: Dict[int, Prompt] = {}
        self.curr_id = 1
        self.curr_points: List[List[float]] = []
        self.curr_labels: List[int] = []
        self.curr_box: Optional[np.ndarray] = None
        self.mode = "point"
        self._finished = False
        self._artists: List[matplotlib.artist.Artist] = []

        self.rs = RectangleSelector(
            ax,
            self.on_select_box,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            interactive=True,
        )
        self.rs.set_active(False)

        fig = ax.figure
        self.cid_click = fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.cid_key = fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.cid_close = fig.canvas.mpl_connect("close_event", self.on_close)
        self.update_title()

    @property
    def finished(self) -> bool:
        return self._finished

    def update_title(self) -> None:
        self.ax.set_title(
            "Object {} | Mode: {} (press 'm' to toggle)\n"
            "Left click: positive, Right click: negative | 'n': next object, 'q': finish".format(
                self.curr_id,
                self.mode.upper(),
            )
        )
        self.ax.figure.canvas.draw_idle()

    def on_select_box(self, eclick, erelease) -> None:
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return
        new_box = np.array([
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        ], dtype=np.float32)
        self.curr_box = new_box
        self.current_box_artist(new_box)
        print(f"Box set for object {self.curr_id}: {new_box.tolist()}")

    def on_click(self, event) -> None:
        if event.inaxes != self.ax:
            return
        if self.mode == "box":
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            label = 1
            color = "green"
        elif event.button == 3:
            label = 0
            color = "red"
        else:
            return
        point = [event.xdata, event.ydata]
        self.curr_points.append(point)
        self.curr_labels.append(label)
        artist = self.ax.scatter(
            point[0],
            point[1],
            color=color,
            marker="*",
            s=200,
            edgecolor="white",
            linewidth=1.0,
        )
        self._artists.append(artist)
        self.ax.figure.canvas.draw_idle()
        print(f"Added {'positive' if label else 'negative'} point for object {self.curr_id}: {point}")

    def on_key(self, event) -> None:
        if event.key == "m":
            self.mode = "box" if self.mode == "point" else "point"
            self.rs.set_active(self.mode == "box")
            self.update_title()
        elif event.key == "n":
            if self.max_objects is not None and self.curr_id >= self.max_objects:
                print("Reached maximum number of objects; ignoring 'n'.")
                return
            self.save_current()
            self.curr_id += 1
            self.reset_current()
            self.update_title()
        elif event.key == "q":
            self.save_current()
            self._finished = True
            plt.close(self.ax.figure)
        elif event.key in {"u", "backspace"}:
            self.undo_last_point()
            self.ax.figure.canvas.draw_idle()

    def undo_last_point(self) -> None:
        if not self.curr_points:
            return
        self.curr_points.pop()
        self.curr_labels.pop()
        for _ in range(1):
            if not self._artists:
                break
            artist = self._artists.pop()
            artist.remove()
        print(f"Removed last point for object {self.curr_id}.")

    def current_box_artist(self, box: np.ndarray) -> None:
        x0, y0, x1, y1 = box.tolist()
        width, height = x1 - x0, y1 - y0
        rect = patches.Rectangle(
            (x0, y0),
            width,
            height,
            edgecolor="cyan",
            facecolor=(0, 0, 0, 0),
            linewidth=2,
        )
        self.ax.add_patch(rect)
        self._artists.append(rect)
        self.ax.figure.canvas.draw_idle()

    def reset_current(self) -> None:
        self.curr_points = []
        self.curr_labels = []
        self.curr_box = None
        self.clear_artists()

    def clear_artists(self) -> None:
        for artist in self._artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self._artists = []
        self.ax.figure.canvas.draw_idle()

    def save_current(self) -> None:
        if not self.curr_points and self.curr_box is None:
            return
        points = np.array(self.curr_points, dtype=np.float32) if self.curr_points else None
        labels = np.array(self.curr_labels, dtype=np.int32) if self.curr_labels else None
        box = np.array(self.curr_box, dtype=np.float32) if self.curr_box is not None else None
        self.objects[self.curr_id] = Prompt(points=points, labels=labels, box=box)
        print(f"Saved object {self.curr_id} with prompts: points={None if points is None else len(points)}, box={'yes' if box is not None else 'no'}")

    def on_close(self, _event) -> None:
        if not self.finished:
            self.save_current()
            self._finished = True

    def get_prompts(self) -> Dict[int, Prompt]:
        self.save_current()
        return self.objects


def select_prompts(image_path: str, max_objects: Optional[int] = None) -> Dict[int, Prompt]:
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(image)
    selector = SAM2Selector(ax, max_objects=max_objects)
    print("Interactive prompt selection controls:\n"
          "  - Left click: positive point\n"
          "  - Right click: negative point\n"
          "  - 'm': toggle box mode (drag to draw box)\n"
          "  - 'n': move to next object\n"
          "  - 'u': or Backspace: undo last point\n"
          "  - 'q': finish and close window")
    plt.show()
    prompts = selector.get_prompts()
    if not prompts:
        raise RuntimeError("No prompts were collected. Please annotate at least one object.")
    return prompts


def apply_prompts_to_predictor(
    predictor,
    inference_state,
    frame_idx: int,
    prompts: Dict[int, Prompt],
) -> Dict[int, np.ndarray]:
    video_masks: Dict[int, np.ndarray] = {}
    for obj_id, prompt in prompts.items():
        points = prompt.points if prompt.points is not None and len(prompt.points) > 0 else None
        labels = prompt.labels if prompt.labels is not None and len(prompt.labels) > 0 else None
        box = prompt.box
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
            box=box,
        )
        for i, out_obj_id in enumerate(out_obj_ids):
            video_masks[out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()
    return video_masks


def preview_prompts(
    image_path: str,
    prompts: Dict[int, Prompt],
    masks: Dict[int, np.ndarray],
    title: str = "Annotated frame",
) -> None:
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(image)
    for obj_id, prompt in prompts.items():
        if prompt.points is not None and prompt.labels is not None:
            show_points(prompt.points, prompt.labels, ax)
        if prompt.box is not None:
            show_box(prompt.box, ax)
    for obj_id, mask in masks.items():
        show_mask(mask, ax, obj_id=obj_id)
    ax.set_title(title)
    plt.show()


def show_mask(mask: np.ndarray, ax: plt.Axes, obj_id: Optional[int] = None, random_color: bool = False) -> None:
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D after squeeze")
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id % 10
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    mask_image = mask[..., None] * color
    ax.imshow(mask_image)


def show_points(coords: np.ndarray, labels: np.ndarray, ax: plt.Axes, marker_size: int = 200) -> None:
    coords = np.asarray(coords)
    labels = np.asarray(labels)
    if coords.size == 0:
        return
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    if len(pos_points) > 0:
        ax.scatter(
            pos_points[:, 0],
            pos_points[:, 1],
            color="green",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )
    if len(neg_points) > 0:
        ax.scatter(
            neg_points[:, 0],
            neg_points[:, 1],
            color="red",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )


def show_box(box: np.ndarray, ax: plt.Axes) -> None:
    x0, y0, x1, y1 = box.tolist()
    width, height = x1 - x0, y1 - y0
    ax.add_patch(
        patches.Rectangle(
            (x0, y0),
            width,
            height,
            edgecolor="green",
            facecolor=(0, 0, 0, 0),
            lw=2,
        )
    )


def propagate_and_export(
    predictor,
    inference_state,
    frame_paths: Sequence[str],
    object_ids: Iterable[int],
    output_dir: str,
    save_overlays: bool,
    save_masks: bool,
    save_tracks: bool,
    overlay_alpha: float,
    supervision_stride: Optional[int] = 20,
    cmap_name: str = "tab10",
) -> TrackingResult:
    os.makedirs(output_dir, exist_ok=True)
    overlay_dir = os.path.join(output_dir, "overlays") if save_overlays else None
    mask_root = os.path.join(output_dir, "masks") if save_masks else None
    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)
    if mask_root:
        os.makedirs(mask_root, exist_ok=True)
    if supervision_stride is not None and supervision_stride <= 0:
        supervision_stride = None
    supervision_dir = None
    if supervision_stride is not None:
        supervision_dir = os.path.join(output_dir, "supervision")
        os.makedirs(supervision_dir, exist_ok=True)
    color_map = plt.get_cmap(cmap_name)
    centroid_tracks: Dict[int, List[Tuple[int, float, float]]] = {obj_id: [] for obj_id in object_ids}
    last_masks: Dict[int, np.ndarray] = {}
    mask_accumulator: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)

    num_frames = len(frame_paths)
    progress = tqdm(total=num_frames, desc="Propagating", dynamic_ncols=True)

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        frame_path = frame_paths[out_frame_idx]
        need_supervision = (
            supervision_dir is not None
            and supervision_stride is not None
            and out_frame_idx % supervision_stride == 0
        )
        need_base_image = (overlay_dir and save_overlays) or need_supervision
        base_image: Optional[np.ndarray] = None
        if need_base_image:
            with Image.open(frame_path) as frame_img:
                base_image = np.array(frame_img.convert("RGB"), dtype=np.float32)
        overlays: Optional[np.ndarray] = (
            base_image.copy() if base_image is not None and overlay_dir and save_overlays else None
        )
        supervision_canvas: Optional[np.ndarray] = (
            base_image.copy() if base_image is not None and need_supervision else None
        )
        for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()
            mask_bool = np.squeeze(mask).astype(bool)
            mask_accumulator[out_obj_id].append((out_frame_idx, mask_bool.copy()))
            if save_masks and mask_root is not None:
                obj_dir = os.path.join(mask_root, f"obj_{out_obj_id:03d}")
                os.makedirs(obj_dir, exist_ok=True)
                mask_img = Image.fromarray(mask_bool.astype(np.uint8) * 255)
                mask_img.save(os.path.join(obj_dir, f"{out_frame_idx:06d}.png"))
            if overlays is not None and overlay_dir is not None:
                color = np.array(color_map(out_obj_id % 10)[:3]) * 255.0
                overlays[mask_bool] = (
                    (1.0 - overlay_alpha) * overlays[mask_bool]
                    + overlay_alpha * color
                )
            if supervision_canvas is not None:
                color = np.array(color_map(out_obj_id % 10)[:3]) * 255.0
                supervision_canvas[mask_bool] = (
                    (1.0 - overlay_alpha) * supervision_canvas[mask_bool]
                    + overlay_alpha * color
                )
            ys, xs = np.where(mask_bool)
            if len(xs) > 0 and len(ys) > 0:
                cx = float(xs.mean())
                cy = float(ys.mean())
                centroid_tracks.setdefault(out_obj_id, []).append((out_frame_idx, cx, cy))
            last_masks[out_obj_id] = mask_bool
        if overlays is not None and overlay_dir is not None:
            overlay_img = Image.fromarray(np.clip(overlays, 0, 255).astype(np.uint8))
            overlay_img.save(os.path.join(overlay_dir, f"{out_frame_idx:06d}.png"))
        if supervision_canvas is not None and supervision_dir is not None:
            sup_img = Image.fromarray(np.clip(supervision_canvas, 0, 255).astype(np.uint8))
            sup_img.save(os.path.join(supervision_dir, f"{out_frame_idx:06d}.png"))
        progress.update(1)
    progress.close()

    if save_tracks:
        for obj_id, samples in centroid_tracks.items():
            if not samples:
                continue
            track_path = os.path.join(output_dir, f"tracks_obj_{obj_id:03d}.csv")
            with open(track_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["frame", "x", "y"])
                writer.writerows(samples)

    for obj_id, samples in mask_accumulator.items():
        if not samples:
            continue
        frame_indices = np.array([idx for idx, _ in samples], dtype=np.int32)
        mask_stack = np.stack([mask.astype(np.uint8) for _, mask in samples], axis=0)
        np.savez_compressed(
            os.path.join(output_dir, f"masks_obj_{obj_id:03d}.npz"),
            frames=frame_indices,
            masks=mask_stack,
        )

    return TrackingResult(centroids=centroid_tracks, last_masks=last_masks)


def serialize_prompts(prompts: Dict[int, Prompt]) -> Dict[int, Dict[str, Optional[List[List[float]]]]]:
    serialized: Dict[int, Dict[str, Optional[List[List[float]]]]] = {}
    for obj_id, prompt in prompts.items():
        serialized[obj_id] = {
            "points": prompt.points.tolist() if prompt.points is not None else None,
            "labels": prompt.labels.tolist() if prompt.labels is not None else None,
            "box": prompt.box.tolist() if prompt.box is not None else None,
        }
    return serialized


def run_sequence(
    predictor,
    frame_paths: Sequence[str],
    output_dir: str,
    interactive: bool,
    interactive_frame_index: int,
    prompts_override: Optional[Dict[int, Prompt]],
    max_objects: Optional[int],
    preview: bool,
    save_overlays: bool,
    save_masks: bool,
    save_tracks: bool,
    overlay_alpha: float,
    supervision_stride: Optional[int],
    flatten_target: Optional[str] = None,
    keep_flattened: bool = False,
) -> Tuple[Dict[int, Prompt], TrackingResult]:
    if not frame_paths:
        raise RuntimeError("No frames available for processing.")
    if interactive_frame_index < 0 or interactive_frame_index >= len(frame_paths):
        raise IndexError(
            f"Frame index {interactive_frame_index} out of range (0, {len(frame_paths) - 1})"
        )

    flatten_dir, cleanup_flat = flatten_frames(
        frame_paths,
        dest_dir=flatten_target,
        keep_dest=keep_flattened,
    )
    print(f"Flattened {len(frame_paths)} frames into {flatten_dir}")

    try:
        inference_state = predictor.init_state(video_path=flatten_dir)
        annotated_frame_path = frame_paths[interactive_frame_index]

        if interactive:
            prompts = select_prompts(annotated_frame_path, max_objects=max_objects)
        else:
            if not prompts_override:
                raise RuntimeError(
                    "Automatic batch chaining requires prompts derived from a previous batch."
                )
            prompts = prompts_override

        masks = apply_prompts_to_predictor(
            predictor,
            inference_state,
            frame_idx=interactive_frame_index,
            prompts=prompts,
        )
        if preview:
            preview_prompts(annotated_frame_path, prompts, masks)

        os.makedirs(output_dir, exist_ok=True)
        metadata_path = os.path.join(output_dir, "prompts.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(serialize_prompts(prompts), f, indent=2)
        print(f"Saved prompt metadata to {metadata_path}")

        tracking_result = propagate_and_export(
            predictor,
            inference_state,
            frame_paths=frame_paths,
            object_ids=prompts.keys(),
            output_dir=output_dir,
            save_overlays=save_overlays,
            save_masks=save_masks,
            save_tracks=save_tracks,
            overlay_alpha=overlay_alpha,
            supervision_stride=supervision_stride,
        )
    finally:
        cleanup_flat()

    return prompts, tracking_result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive SAM 2 tracking helper")
    parser.add_argument("--frame-root", required=True, help="Root directory containing frames or batch_* subfolders")
    parser.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_large.pt", help="Path to SAM 2 checkpoint")
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml", help="Path to SAM 2 config")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--batch-pattern", default="batch_*", help="Glob for batch subdirectories")
    parser.add_argument("--interactive-frame-index", type=int, default=0, help="Frame index to annotate")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on total frames to process")
    parser.add_argument("--flatten-dir", default=None, help="Optional directory to place flattened frames")
    parser.add_argument("--keep-flattened", action="store_true", help="Keep flattened frames instead of deleting temp directory")
    parser.add_argument("--max-objects", type=int, default=None, help="Optional limit on objects to annotate")
    parser.add_argument("--output-dir", default=None, help="Directory to store tracking outputs")
    parser.add_argument("--save-overlays", action="store_true", help="Save RGB overlays for each frame")
    parser.add_argument("--save-masks", action="store_true", help="Save binary mask PNGs for each object and frame")
    parser.add_argument("--save-tracks", action="store_true", help="Export centroid tracks as CSV files")
    parser.add_argument("--overlay-alpha", type=float, default=0.5, help="Alpha value for mask overlays")
    parser.add_argument("--preview", action="store_true", help="Preview masks on annotated frame before propagation")
    parser.add_argument("--supervision-stride", type=int, default=20, help="Stride (in frames) for saving supervision overlays; set to 0 to disable")
    parser.add_argument("--chain-batches", action="store_true", help="Process contiguous batch_* folders sequentially")
    parser.add_argument("--chain-tolerance", type=int, default=5, help="Pixel padding applied to derived bounding boxes when chaining batches")
    parser.add_argument("--roi-file", default=None, help="Optional ROI prompts JSON to load instead of opening the GUI")
    parser.add_argument("--roi-name", default="roi_prompts.json", help="Filename to search for ROI prompts under frame-root/output-dir")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_matplotlib()

    device = resolve_device(args.device)
    configure_torch(device)
    predictor = build_sam2_video_predictor(args.config, args.checkpoint, device=device)
    combined_rows: List[Tuple[int, int, float, float]] = []

    if args.chain_batches:
        if args.max_frames is not None:
            print("Warning: --max-frames is ignored when chaining batches.")
        if args.flatten_dir is not None:
            print("Warning: --flatten-dir is ignored when chaining batches.")
        if args.keep_flattened:
            print("Warning: --keep-flattened is ignored when chaining batches.")

        batch_dirs = discover_contiguous_batches(args.frame_root, args.batch_pattern)
        if not batch_dirs:
            raise RuntimeError("No batch folders found to process.")

        global_offset = 0
        prompts_for_next: Optional[Dict[int, Prompt]] = None

        for batch_idx, batch_dir in enumerate(batch_dirs):
            frame_paths = collect_frame_paths(
                root=batch_dir,
                batch_pattern=args.batch_pattern,
                max_frames=None,
            )
            if not frame_paths:
                print(f"Skipping empty batch folder: {batch_dir}")
                continue

            interactive = batch_idx == 0
            frame_index = args.interactive_frame_index if interactive else 0
            batch_name = os.path.basename(os.path.normpath(batch_dir))

            if args.output_dir:
                batch_output_dir = os.path.join(os.path.abspath(args.output_dir), batch_name)
            else:
                batch_output_dir = os.path.join(os.path.abspath(batch_dir), "sam2_tracking")

            prompts_override: Optional[Dict[int, Prompt]] = None
            if interactive:
                roi_path = find_roi_prompts_file(
                    frame_root=batch_dir,
                    output_dir=batch_output_dir,
                    roi_file=args.roi_file,
                    roi_name=args.roi_name,
                )
                if roi_path:
                    prompts_override = load_prompts_json(roi_path)
                    interactive = False
                    print(f"Loaded ROI prompts from {roi_path}")

            _, tracking_result = run_sequence(
                predictor=predictor,
                frame_paths=frame_paths,
                output_dir=batch_output_dir,
                interactive=interactive,
                interactive_frame_index=frame_index,
                prompts_override=prompts_override if prompts_override is not None else prompts_for_next,
                max_objects=args.max_objects,
                preview=args.preview if interactive else False,
                save_overlays=args.save_overlays,
                save_masks=args.save_masks,
                save_tracks=args.save_tracks,
                overlay_alpha=args.overlay_alpha,
                supervision_stride=args.supervision_stride,
                flatten_target=None,
                keep_flattened=False,
            )

            for obj_id, samples in tracking_result.centroids.items():
                for frame_idx, cx, cy in samples:
                    combined_rows.append((frame_idx + global_offset, obj_id, cx, cy))
            global_offset += len(frame_paths)

            prompts_for_next = derive_prompts_from_masks(
                tracking_result.last_masks,
                tolerance_px=args.chain_tolerance,
            )
            if not prompts_for_next:
                print(f"Warning: no prompts derived from {batch_name}; stopping batch chaining.")
                break

        combined_rows.sort(key=lambda item: (item[0], item[1]))
        print("Batch chaining complete.")
    else:
        frame_paths = collect_frame_paths(
            root=args.frame_root,
            batch_pattern=args.batch_pattern,
            max_frames=args.max_frames,
        )

        output_dir = (
            os.path.abspath(args.output_dir)
            if args.output_dir is not None
            else os.path.join(os.path.abspath(args.frame_root), "sam2_tracking")
        )

        prompts_override: Optional[Dict[int, Prompt]] = None
        roi_path = find_roi_prompts_file(
            frame_root=args.frame_root,
            output_dir=output_dir,
            roi_file=args.roi_file,
            roi_name=args.roi_name,
        )
        if roi_path:
            prompts_override = load_prompts_json(roi_path)
            print(f"Loaded ROI prompts from {roi_path}")

        _, tracking_result = run_sequence(
            predictor=predictor,
            frame_paths=frame_paths,
            output_dir=output_dir,
            interactive=prompts_override is None,
            interactive_frame_index=args.interactive_frame_index,
            prompts_override=prompts_override,
            max_objects=args.max_objects,
            preview=args.preview,
            save_overlays=args.save_overlays,
            save_masks=args.save_masks,
            save_tracks=args.save_tracks,
            overlay_alpha=args.overlay_alpha,
            supervision_stride=args.supervision_stride,
            flatten_target=args.flatten_dir,
            keep_flattened=args.keep_flattened,
        )

        for obj_id, samples in tracking_result.centroids.items():
            for frame_idx, cx, cy in samples:
                combined_rows.append((frame_idx, obj_id, cx, cy))
        combined_rows.sort(key=lambda item: (item[0], item[1]))
        print("Tracking complete for single sequence.")

    parent_dir = os.path.dirname(os.path.normpath(os.path.abspath(args.frame_root)))
    folder_name = os.path.basename(os.path.normpath(args.frame_root))
    csv_name = f"{folder_name}.csv"
    csv_output_path = os.path.join(parent_dir, csv_name)
    with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "object_id", "centroid_x", "centroid_y"])
        writer.writerows(combined_rows)
    print(f"Saved combined track CSV to {csv_output_path}")

    print("Tracking complete.")


if __name__ == "__main__":
    main()

    """
     /home/d25u2/anaconda3/envs/torch3.11/bin/python Dynamics/dynamics.py \
        --frame-root /media/d25u2/Dont/Duffing/sam-2/Dynamics/T_05/batch_000 \
        --device cuda \


     /home/d25u2/anaconda3/envs/torch3.11/bin/python Dynamics/dynamics.py \
        --frame-root /media/d25u2/Dont/Duffing/sam-2/Dynamics/T_05/batch_000 \
        --device cuda \
        --chain-batches --save-tracks --chain-tolerance 5


     /home/d25u2/anaconda3/envs/torch3.11/bin/python /media/d25u2/Dont/Duffing/sam-2/Dynamics/dynamics.py   --frame-root //"media/d25u2/Dont/Duffing/shaker project new data_compressed/dataset1/Conv_2.2_2_3/batch_000"   --device cuda   --save-masks --chain-batches --save-tracks --chain-tolerance 5   --supervision-stride 20
    """
