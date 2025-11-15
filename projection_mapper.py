import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict


class PointPicker:
    def __init__(self, window_name: str, image: np.ndarray, prompt: str = "", max_points: int = 4):
        self.window_name = window_name
        self.image = image.copy()
        self.display = image.copy()
        self.prompt = prompt
        self.max_points = max_points
        self.points: List[Tuple[int, int]] = []
        self.done = False
        self.canceled = False

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.done:
            if len(self.points) < self.max_points:
                self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and not self.done:
            if self.points:
                self.points.pop()

    def pick(self) -> List[Tuple[int, int]]:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_cb)
        while True:
            self.display = self.image.copy()
            # Draw prompt and points
            if self.prompt:
                cv2.rectangle(self.display, (0, 0), (self.display.shape[1], 36), (0, 0, 0), -1)
                cv2.putText(self.display, self.prompt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for i, (px, py) in enumerate(self.points):
                cv2.circle(self.display, (px, py), 5, (0, 255, 0), -1)
                cv2.putText(self.display, f"{i+1}", (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if i > 0:
                    cv2.line(self.display, self.points[i-1], self.points[i], (0, 255, 0), 2)
            if len(self.points) == self.max_points and self.max_points >= 3:
                cv2.line(self.display, self.points[-1], self.points[0], (0, 255, 0), 2)

            cv2.imshow(self.window_name, self.display)
            key = cv2.waitKey(16) & 0xFF

            if key in (13, 10):  # Enter -> confirm
                if len(self.points) == self.max_points:
                    self.done = True
                    break
            elif key == 27:  # ESC -> cancel
                self.canceled = True
                break
            elif key in (8, 127):  # Backspace/Delete -> undo
                if self.points:
                    self.points.pop()

        cv2.setMouseCallback(self.window_name, lambda *args: None)
        return self.points if not self.canceled else []


def order_quad(points: List[Tuple[int, int]]) -> np.ndarray:
    pts = np.array(points, dtype=np.float32)
    # Order as: top-left, top-right, bottom-right, bottom-left
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


class FaceMapping:
    def __init__(self, name: str):
        self.name = name
        self.src_pts: List[Tuple[int, int]] = []
        self.dst_pts: List[Tuple[int, int]] = []
        self.H: np.ndarray | None = None
        self.mask: np.ndarray | None = None

    def compute(self, canvas_shape: Tuple[int, int]):
        if len(self.src_pts) == 4 and len(self.dst_pts) == 4:
            src = order_quad(self.src_pts)
            dst = order_quad(self.dst_pts)
            self.H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            mask = np.zeros((canvas_shape[1], canvas_shape[0]), dtype=np.uint8)
            cv2.fillConvexPoly(mask, dst.astype(np.int32), 255)
            self.mask = mask
        else:
            self.H = None
            self.mask = None

    def to_json(self):
        return {
            "name": self.name,
            "src_pts": self.src_pts,
            "dst_pts": self.dst_pts,
        }

    @staticmethod
    def from_json(d):
        fm = FaceMapping(d["name"])
        fm.src_pts = [tuple(map(int, p)) for p in d["src_pts"]]
        fm.dst_pts = [tuple(map(int, p)) for p in d["dst_pts"]]
        return fm


class ProjectionMapperApp:
    def __init__(self, cam_index: int = 0, canvas_w: int = 1280, canvas_h: int = 720, presets_path: str = "mapping_presets.json"):
        self.cam_index = cam_index
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.face_names = ["left", "front", "right"]
        self.faces: Dict[str, FaceMapping] = {n: FaceMapping(n) for n in self.face_names}
        self.cap = None
        self.fullscreen = False
        self.last_frame = None
        self.overlay_grid = False
        self._cam_switch_locked = False
        self.presets_path = presets_path

    def _open_cam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap.isOpened():
            raise RuntimeError("Camera open failed")

    def _read_frame(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read from camera")
        self.last_frame = frame
        return frame

    def _set_cam(self, idx: int) -> bool:
        prev_idx = self.cam_index
        prev_cap = self.cap
        if self.cap:
            self.cap.release()
        self.cap = None
        self.cam_index = idx
        try:
            self._open_cam()
            return True
        except Exception as e:
            print(f"Failed to switch camera to index {idx}: {e}")
            self.cam_index = prev_idx
            self.cap = None
            try:
                # Restore previous camera
                self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            except Exception:
                pass
            return False

    def _blank_canvas(self) -> np.ndarray:
        return np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

    def _draw_help(self, img: np.ndarray, text_lines: List[str], y0: int = 10):
        x = 10
        y = y0
        for t in text_lines:
            cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 24

    def _toggle_fullscreen(self, window: str, enable: bool):
        prop = cv2.WINDOW_FULLSCREEN if enable else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, prop)

    def _preview_and_select_snapshot(self, window_name: str, prompt_lines: List[str]) -> np.ndarray | None:
        try:
            self._open_cam()
        except Exception as e:
            print(f"Camera open failed in preview: {e}")
            return None
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        snap = None
        while True:
            ok, frame = (self.cap.read() if self.cap is not None else (False, None))
            if not ok or frame is None:
                # Try reopen current cam quickly
                try:
                    if self.cap:
                        self.cap.release()
                    self.cap = None
                    self._open_cam()
                    ok, frame = self.cap.read()
                except Exception:
                    ok = False
            if ok and frame is not None:
                disp = frame.copy()
                help_lines = list(prompt_lines)
                help_lines.append(f"Camera index: {self.cam_index} | n: next camera | Enter: snapshot | ESC: cancel")
                self._draw_help(disp, help_lines, y0=24)
                cv2.imshow(window_name, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (13, 10):  # Enter -> use current frame
                if ok and frame is not None:
                    snap = frame.copy()
                    break
            elif key == 27:  # ESC -> cancel
                snap = None
                break
            elif key == ord('n'):
                # Cycle to next camera index; do not set any locks here
                self._set_cam(self.cam_index + 1)
        cv2.destroyWindow(window_name)
        return snap

    def calibrate_face(self, face: FaceMapping):
        # Destination points on canvas
        canvas = self._blank_canvas()
        cv2.namedWindow("Projector Output", cv2.WINDOW_NORMAL)
        if self.fullscreen:
            self._toggle_fullscreen("Projector Output", True)
        self._draw_help(canvas, [
            f"Select 4 destination points for '{face.name}' on the canvas (order around the quad).",
            "Left-click: add point | Right-click: undo | Enter: confirm | ESC: cancel",
        ], y0=30)
        picker_dst = PointPicker("Projector Output", canvas, prompt=f"Pick 4 dst points: {face.name}")
        dst_pts = picker_dst.pick()
        if len(dst_pts) != 4:
            return False
        face.dst_pts = dst_pts

        # Source points on camera snapshot
        snap = self._preview_and_select_snapshot(
            window_name="Camera Preview",
            prompt_lines=[
                f"Select camera for '{face.name}': Enter to use snapshot | n: next camera | ESC: cancel",
            ],
        )
        if snap is None:
            return False
        disp = snap.copy()
        self._draw_help(disp, [
            f"Select 4 source points from camera for '{face.name}' (corresponding order)",
            "Left-click: add point | Right-click: undo | Enter: confirm | ESC: cancel",
        ], y0=30)
        picker_src = PointPicker("Camera Snapshot", disp, prompt=f"Pick 4 src points: {face.name}")
        src_pts = picker_src.pick()
        cv2.destroyWindow("Camera Snapshot")
        if len(src_pts) != 4:
            return False
        face.src_pts = src_pts

        face.compute((self.canvas_w, self.canvas_h))
        return face.H is not None

    def calibrate_all(self):
        for name in self.face_names:
            ok = self.calibrate_face(self.faces[name])
            if not ok:
                print(f"Calibration canceled or failed for {name}")
                return False
        print("Calibration done for all faces.")
        return True

    def save_config(self, path: str):
        data = {n: self.faces[n].to_json() for n in self.face_names}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved config to {path}")

    def load_config(self, path: str):
        if not os.path.exists(path):
            print(f"Config not found: {path}")
            return False
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for n in self.face_names:
            if n in data:
                self.faces[n] = FaceMapping.from_json(data[n])
        for n in self.face_names:
            self.faces[n].compute((self.canvas_w, self.canvas_h))
        print(f"Loaded config from {path}")
        return True

    def save_preset(self, name: str, path: str | None = None):
        path = path or self.presets_path
        presets = {}
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    presets = json.load(f)
            except Exception:
                presets = {}
        presets[name] = {n: self.faces[n].to_json() for n in self.face_names}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(presets, f, ensure_ascii=False, indent=2)
        print(f"Saved preset '{name}' to {path}")

    def load_preset(self, name: str, path: str | None = None) -> bool:
        path = path or self.presets_path
        if not os.path.exists(path):
            print(f"Presets file not found: {path}")
            return False
        with open(path, 'r', encoding='utf-8') as f:
            presets = json.load(f)
        if name not in presets:
            print(f"Preset not found: '{name}'")
            return False
        data = presets[name]
        for n in self.face_names:
            if n in data:
                self.faces[n] = FaceMapping.from_json(data[n])
        for n in self.face_names:
            self.faces[n].compute((self.canvas_w, self.canvas_h))
        print(f"Loaded preset '{name}' from {path}")
        return True

    def _draw_grid(self, img: np.ndarray, step: int = 80, color=(60, 60, 60)):
        h, w = img.shape[:2]
        for x in range(0, w, step):
            cv2.line(img, (x, 0), (x, h), color, 1)
        for y in range(0, h, step):
            cv2.line(img, (0, y), (w, y), color, 1)

    def run(self, config_path: str = "mapping_config.json"):
        self._open_cam()
        window = "Projector Output"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            self._toggle_fullscreen(window, True)

        print("Controls: q: quit | f: fullscreen toggle | g: grid | c: recalibrate all | s: save | l: load")
        print("Tip: Click the projector window before selecting points.")

        # Initial calibration
        if not any(self.faces[n].H is not None for n in self.face_names):
            self.calibrate_all()

        while True:
            frame = self._read_frame()

            canvas = self._blank_canvas()
            if self.overlay_grid:
                self._draw_grid(canvas)

            # Compose each face
            for n in self.face_names:
                face = self.faces[n]
                if face.H is not None and face.mask is not None:
                    warped = cv2.warpPerspective(frame, face.H, (self.canvas_w, self.canvas_h))
                    mask3 = cv2.merge([face.mask, face.mask, face.mask])
                    inv_mask3 = cv2.bitwise_not(mask3)
                    canvas = cv2.bitwise_and(canvas, inv_mask3)
                    canvas = cv2.add(canvas, cv2.bitwise_and(warped, mask3))

            # Heads-up help overlay
            help_lines = [
                "q: quit  f: fullscreen  g: grid  c: calibrate  s: save  l: load  S: save preset  L: load preset",
                "Faces: left, front, right | Use corresponding feature points in the same order",
            ]
            if not self._cam_switch_locked:
                help_lines.append(f"Camera index: {self.cam_index}  |  n: select next camera (once)")
            else:
                help_lines.append(f"Camera index: {self.cam_index}  |  camera selection locked")
            self._draw_help(canvas, help_lines, y0=24)

            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('f'):
                self.fullscreen = not self.fullscreen
                self._toggle_fullscreen(window, self.fullscreen)
            elif key == ord('g'):
                self.overlay_grid = not self.overlay_grid
            elif key == ord('c'):
                self.calibrate_all()
            elif key == ord('s'):
                self.save_config(config_path)
            elif key == ord('l'):
                self.load_config(config_path)
            elif key == ord('S'):
                try:
                    preset_name = input("Preset name to save: ").strip()
                except EOFError:
                    preset_name = ""
                if preset_name:
                    self.save_preset(preset_name, self.presets_path)
            elif key == ord('L'):
                try:
                    preset_name = input("Preset name to load: ").strip()
                except EOFError:
                    preset_name = ""
                if preset_name:
                    self.load_preset(preset_name, self.presets_path)
            elif key == ord('n') and not self._cam_switch_locked:
                if self._set_cam(self.cam_index + 1):
                    self._cam_switch_locked = True

        cv2.destroyAllWindows()
        if self.cap:
            self.cap.release()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simple 3-face projection mapper (left, front, right)")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--w", type=int, default=1280, help="Projector canvas width")
    parser.add_argument("--h", type=int, default=720, help="Projector canvas height")
    parser.add_argument("--config", type=str, default="mapping_config.json", help="Config file path")
    parser.add_argument("--preset", type=str, default="", help="Preset name to auto-load on startup")
    parser.add_argument("--presets", type=str, default="mapping_presets.json", help="Presets file path")
    args = parser.parse_args()

    app = ProjectionMapperApp(cam_index=args.cam, canvas_w=args.w, canvas_h=args.h, presets_path=args.presets)
    if args.preset:
        app.load_preset(args.preset, args.presets)
    app.run(config_path=args.config)
