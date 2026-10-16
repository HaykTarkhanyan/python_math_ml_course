"""Play Snake with your thumb: the trained gesture model steers, the webcam watches.

Show a thumb in the green box: up / down / left / right turns the snake, "nothing" keeps it
going straight. The walls wrap around (webcam control has lag, so walls would be cruel); only
biting yourself ends the game.

A single frame's prediction is noisy, so the snake turns only when at least TURN_VOTES of the
last TURN_WINDOW confident predictions agree - a moving vote over the prediction stream.

Keys: W/A/S/D also steer (always on), R = restart, B = move the box, Q or Esc = quit.

Run (after the notebook exported gesture_model.pt2 next to this file):
    python play_snake.py
    python play_snake.py --keyboard          # no model, no camera: just the game, WASD
Needs opencv-python, numpy, torch (tested with opencv-python 4.11.0.86, numpy 1.26.4,
torch 2.9.0).
"""
import argparse
import logging
import random
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

from gesture_common import CLASSES, CROP_SIZE, bgr_to_tensor, box_coords, crop_box, mirror

GRID = 16            # board is GRID x GRID cells
CELL = 30            # pixels per cell -> a 480 px board
TICK_SECONDS = 0.3   # one snake step every TICK_SECONDS
TURN_WINDOW = 5      # look at the last TURN_WINDOW predictions...
TURN_VOTES = 4       # ...and turn when TURN_VOTES of them agree on a direction
MIN_CONFIDENCE = 0.6  # predictions below this count as "nothing"
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
KEYS = {ord("w"): "up", ord("s"): "down", ord("a"): "left", ord("d"): "right"}
HERE = Path(__file__).resolve().parent
WINDOW = "gesture snake - R restart, B move box, Q quit"

log = logging.getLogger("play_snake")


class Snake:
    """The game, with no drawing and no camera, so it can be tested on its own."""

    def __init__(self, grid=GRID, seed=509):
        self.grid, self.rng = grid, random.Random(seed)
        mid = grid // 2
        self.body = deque([(mid, mid - 2), (mid, mid - 1), (mid, mid)])  # tail ... head
        self.heading, self.moved = "right", "right"
        self.alive, self.score = True, 0
        self.food = self._new_food()

    def _new_food(self):
        free = [(r, c) for r in range(self.grid) for c in range(self.grid) if (r, c) not in self.body]
        return self.rng.choice(free)

    def turn(self, direction):
        # compare with the direction of the LAST STEP, so two quick turns cannot reverse the snake
        if direction in MOVES and direction != OPPOSITE[self.moved]:
            self.heading = direction

    def step(self):
        if not self.alive:
            return
        dr, dc = MOVES[self.heading]
        r, c = self.body[-1]
        head = ((r + dr) % self.grid, (c + dc) % self.grid)
        self.moved = self.heading
        grows = head == self.food
        if head in list(self.body)[0 if grows else 1:]:  # the tail cell frees up unless we grow
            self.alive = False
            return
        self.body.append(head)
        if grows:
            self.score += 1
            self.food = self._new_food()
        else:
            self.body.popleft()


class Smoother:
    """Turns a noisy stream of per-frame predictions into occasional, deliberate turns."""

    def __init__(self, window=TURN_WINDOW, votes=TURN_VOTES, min_conf=MIN_CONFIDENCE):
        self.recent, self.votes, self.min_conf = deque(maxlen=window), votes, min_conf

    def update(self, label, confidence):
        self.recent.append(label if confidence >= self.min_conf else "nothing")
        top, n = Counter(self.recent).most_common(1)[0]
        return top if top in MOVES and n >= self.votes else None


def load_model(path):
    import torch
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the notebook's export cell first, "
                                "or play with --keyboard")
    model = torch.export.load(str(path)).module()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, CROP_SIZE, CROP_SIZE))
    if tuple(out.shape) != (1, len(CLASSES)):
        raise ValueError(f"model returns shape {tuple(out.shape)}, expected (1, {len(CLASSES)}) "
                         f"logits in the order {CLASSES}")
    log.info(f"loaded {path}")
    return model


def predict(model, crop):
    import torch
    with torch.no_grad():
        p = torch.softmax(model(bgr_to_tensor(crop)), dim=1)[0].numpy()
    return CLASSES[int(p.argmax())], float(p.max()), p


def draw_board(game):
    board = np.full((GRID * CELL, GRID * CELL, 3), 30, np.uint8)
    fr, fc = game.food
    cv2.circle(board, (fc * CELL + CELL // 2, fr * CELL + CELL // 2), CELL // 3, (60, 60, 230), -1)
    for i, (r, c) in enumerate(game.body):
        shade = (0, 255, 120) if i == len(game.body) - 1 else (0, 170, 80)
        cv2.rectangle(board, (c * CELL + 1, r * CELL + 1), ((c + 1) * CELL - 2, (r + 1) * CELL - 2), shade, -1)
    cv2.putText(board, f"score {game.score}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if not game.alive:
        mid = GRID * CELL // 2
        cv2.rectangle(board, (0, mid - 32), (board.shape[1], mid + 14), (0, 0, 0), -1)
        cv2.putText(board, "GAME OVER - R to restart", (40, mid),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 255), 2)
    return board


def draw_camera(frame, box, label, conf, probs, fps):
    x0, y0, s = box
    view = frame.copy()
    panel = view[:50 + 22 * len(CLASSES), :250]
    panel //= 3  # darken in place, so the text reads on a white wall too
    cv2.rectangle(view, (x0, y0), (x0 + s, y0 + s), (0, 200, 0), 3)
    cv2.putText(view, f"{label} {conf:.2f}   {fps:.0f} fps", (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2)
    for i, (c, p) in enumerate(zip(CLASSES, probs)):
        y = 50 + 22 * i
        cv2.rectangle(view, (90, y - 12), (90 + int(150 * p), y + 4), (0, 200, 255), -1)
        cv2.putText(view, c, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    scale = GRID * CELL / view.shape[0]
    return cv2.resize(view, (int(view.shape[1] * scale), GRID * CELL))


def setup_logging():
    Path("logs").mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler("logs/play_snake.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=HERE / "gesture_model.pt2")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--keyboard", action="store_true", help="no model and no camera, WASD only")
    args = ap.parse_args()
    setup_logging()

    model = cap = None
    if not args.keyboard:
        model = load_model(args.model)
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera {args.camera}; close other webcam apps or try --camera 1")
    game, smoother, side = Snake(seed=int(time.time())), Smoother(), "right"
    next_tick, fps, t_prev = time.monotonic() + TICK_SECONDS, 0.0, time.monotonic()
    cv2.namedWindow(WINDOW)
    try:
        while True:
            panels = []
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("the camera stopped returning frames")
                frame = mirror(frame)
                box = box_coords(frame.shape, side)
                label, conf, probs = predict(model, crop_box(frame, box))
                direction = smoother.update(label, conf)
                if direction:
                    game.turn(direction)
                now = time.monotonic()
                fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
                t_prev = now
                panels.append(draw_camera(frame, box, label, conf, probs, fps))
            if time.monotonic() >= next_tick:
                was_alive = game.alive
                game.step()
                if was_alive and not game.alive:
                    log.info(f"game over, score {game.score}")
                next_tick = time.monotonic() + TICK_SECONDS  # never "catch up" after a stall
            panels.append(draw_board(game))
            cv2.imshow(WINDOW, np.hstack(panels))
            key = cv2.waitKey(1 if cap is not None else 15) & 0xFF
            if ord("A") <= key <= ord("Z"):
                key += 32
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in KEYS:
                game.turn(KEYS[key])
            elif key == ord("r"):
                game, next_tick = Snake(seed=int(time.time())), time.monotonic() + TICK_SECONDS
            elif key == ord("b"):
                side = "left" if side == "right" else "right"
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
