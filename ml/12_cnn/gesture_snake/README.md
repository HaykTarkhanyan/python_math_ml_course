# Gesture Snake - steer Snake with your thumb

Record your own thumb gestures with the webcam, train a CNN that reads which way the thumb points
(up, down, left, right, or nothing), and play Snake with it. Everything runs on a laptop CPU.

| File | What it is |
|---|---|
| `record_gestures.py` | records your dataset from the webcam |
| `gesture_snake.ipynb` | the practical: fill in the `# YOUR CODE HERE` cells, train, export `gesture_model.pt2` |
| `gesture_snake_solution.ipynb` | the same notebook, solved |
| `play_snake.py` | the game: webcam + your model |
| `gesture_common.py` | shared by all three: classes, the box, the crop, the image format |
| `web_sample/` | 90 photos of strangers from the HaGRID dataset (up / down / nothing) |

## 1. Install (once)

Tested with Python 3.11.

```
pip install -r requirements.txt
```

On Linux without an NVIDIA GPU, first install the CPU build of PyTorch (much smaller):

```
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 2. Record your data (about 3 minutes)

```
python record_gestures.py --user yourname
```

Put your thumb in the green box and press a key. Each press records one **burst**: a 2-second
countdown, then 50 frames over about 5 seconds.

| Key | Records |
|---|---|
| W / S / A / D | thumb up / down / left / right |
| N | nothing: an empty box, a fist, an open palm |
| B | moves the box to the other side of the screen |
| Q | quits |

- Record **at least 2 bursts per class** - the notebook holds one burst out to test on.
- During a burst move a little: nearer, farther, slightly tilted.
- Between bursts change something: where you sit, a lamp on or off.
- Directions are **yours**. The preview is mirrored like a selfie camera, so thumb-left looks
  like thumb-left on the screen.

The crops land in `gesture_data/yourname/<class>/`.

## 3. Train

Open `gesture_snake.ipynb` in VS Code or Jupyter, fill in the `# YOUR CODE HERE` cells (the asserts
tell you when a piece works) and run it top to bottom. The last cell writes `gesture_model.pt2`:
whichever of your two models scored higher on the held-out bursts.

**No time on your laptop?** Zip the whole folder (with `gesture_data/`), upload it to Google Colab,
unzip it there, run the notebook, and download `gesture_model.pt2` back into this folder. The game
itself has to run on your laptop: Colab cannot show a live webcam.

## 4. Play

```
python play_snake.py
```

Show a thumb in the green box to turn; "nothing" keeps the snake going straight. The walls wrap
around; only biting yourself ends the game. R restarts, Q quits. W/A/S/D work too.

To try the game without a model or camera: `python play_snake.py --keyboard`.

## When something goes wrong

- **"could not open camera"** - close Zoom, Teams or any browser tab that uses the camera; or try
  `--camera 1`.
- **macOS** - allow camera access for Terminal (or VS Code) in System Settings > Privacy & Security.
- **The snake ignores you or turns late** - it turns only when 4 of the last 5 confident predictions
  agree. Check the bars on the left of the game window: if they flicker, record more bursts in the
  place and light where you play.
