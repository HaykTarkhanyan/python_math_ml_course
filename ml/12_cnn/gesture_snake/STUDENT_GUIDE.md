# Gesture Snake - student guide

Record your own hand gestures with the webcam, train any model you like on them, and steer a Snake
game with your thumb. The three Python files are small and meant to be reused in your own projects.

| File | What it does |
|---|---|
| `record_gestures.py` | records a dataset of your gestures from the webcam |
| `play_snake.py` | plays Snake, steered by a trained model watching the webcam |
| `gesture_common.py` | shared by both: the class list, the box on the screen, the crop, the image format |

Keep the three files in one folder: both scripts import `gesture_common.py`.

---

## 1. Install

Tested with Python 3.11. Recording needs only OpenCV and NumPy:

```
pip install opencv-python==4.11.0.86 numpy==1.26.4
```

Training and playing also need PyTorch (the full list is in `requirements.txt`):

```
pip install torch==2.9.0 torchvision==0.24.0
```

On Linux without an NVIDIA GPU, add `--index-url https://download.pytorch.org/whl/cpu` to the
PyTorch line, or pip downloads the much larger CUDA build, which you cannot use.

---

## 2. Record your data (about 3 minutes)

```
python record_gestures.py --user yourname
```

A window opens with your webcam and a green box. Click the window so it receives your keys, put
your hand in the box, and press:

| Key | Records |
|---|---|
| W | thumb up |
| S | thumb down |
| A | thumb pointing to **your** left |
| D | thumb pointing to **your** right |
| N | nothing: an empty box, a fist, an open palm, a hand passing through |
| B | moves the box to the other side of the screen |
| Q | quits |

Each key press records one **burst**: a 2-second countdown, then 50 frames over about 5 seconds.
The box turns red while it records. The counts on the left show what you have so far.

What gets saved:

- **only the inside of the box** - no full frames, no video, no sound;
- as a 128 x 128 JPEG of about 5 KB, mirrored like a selfie (thumb-left points left in the image);
- in `gesture_data/yourname/<class>/yourname_<burst>_<nn>.jpg`. The burst id is a timestamp.

Running the recorder again **adds** to what is there. To start over, delete
`gesture_data/yourname/`.

**How to record well**

- Record **at least 2 bursts per class**, better 3-4. Frames inside one burst are a tenth of a
  second apart - near-copies - so one burst is closer to *one* example than to 50.
- **Change something between bursts**: where you sit, a lamp on or off, your sleeve, the box side
  (B). Move a little during a burst too: nearer, farther, slightly tilted.
- **Watch the background.** Whatever else is inside the box - a radiator, a red sleeve that only
  appears in "down" - is something a model can learn *instead of* your thumb. Vary it, or keep it
  on purpose and use Grad-CAM to catch the model cheating.

---

## 3. Train a model

The practical notebook (`gesture_snake_solution.ipynb`) trains two models on `gesture_data/` and
writes `gesture_model.pt2`. You can also train your own, any way you like, as long as the exported
model keeps this contract:

| | |
|---|---|
| input | a float tensor of shape `(N, 3, 128, 128)`: RGB, values in `[0, 1]` |
| output | `(N, 5)` logits, in the order of `CLASSES` in `gesture_common.py`: up, down, left, right, nothing |

Anything else your model needs - resizing, ImageNet normalization - goes **inside** the model.

`gesture_common.bgr_to_tensor(images)` turns saved crops (or webcam crops) into exactly that input.

**Two things that will bite you**

- **Validate on whole bursts.** If you split frames at random, the validation frames have
  near-copies in training and the score measures memory. A nearest-neighbour model with no
  understanding at all scores near 100% that way.
- **A mirror flip changes the label.** Flipping is a fine augmentation, but a flipped "left" is a
  "right". Swap the label when you flip.

**Export** with `torch.export`, so the game can load the model without your Python class:

```python
import torch

model.eval()
example = torch.zeros(2, 3, 128, 128)   # 2, not 1: with 1, export freezes the batch size
exported = torch.export.export(model, (example,), dynamic_shapes=({0: torch.export.Dim.DYNAMIC},))
torch.export.save(exported, "gesture_model.pt2")
```

---

## 4. Play

```
python play_snake.py                          # uses gesture_model.pt2 next to the script
python play_snake.py --model my_model.pt2
python play_snake.py --keyboard               # no model, no camera: just the game
```

Show a thumb in the green box to turn. "nothing" keeps the snake going straight. The walls wrap
around; only biting yourself ends the game. W/A/S/D also steer. R restarts, B moves the box, Q quits.

The bars on the left of the window are the model's probabilities for the current frame.
A single frame is noisy, so the snake turns only when **4 of the last 5** predictions agree on a
direction with at least 60% confidence. A snake that ignores you or turns late usually means the
bars flicker: record more bursts in the place and light where you play.

Record and play in the same spot, with the box on the same side: the model only knows what you
showed it.

---

## 5. Reuse the pieces

**Different classes.** Edit `CLASSES` in `gesture_common.py` and the `KEYS` dictionary at the top of
`record_gestures.py` (key -> class name). The recorder then records your classes - rock / paper /
scissors, a number of fingers, anything a hand can show. In the game only up / down / left / right
steer; any other class acts like "nothing".

**The box.** `BOX_FRAC` (box side as a share of the frame height) and `BOX_MARGIN` in
`gesture_common.py`. Change them and re-record: a model trained on one box size sees a different
picture in another.

**A different camera.** `--camera 1` on either script.

**Your own program.** The model works anywhere you can read a webcam frame:

```python
import cv2, torch
from gesture_common import CLASSES, bgr_to_tensor, box_coords, crop_box, mirror

model = torch.export.load("gesture_model.pt2").module()
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
frame = mirror(frame)                                   # the same selfie view as when recording
crop = crop_box(frame, box_coords(frame.shape))         # the same box
with torch.no_grad():
    probs = torch.softmax(model(bgr_to_tensor(crop)), dim=1)[0]
print(CLASSES[int(probs.argmax())], float(probs.max()))
```

Use the same `mirror`, `box_coords` and `crop_box` as the recorder. A model fed a different crop
than it was trained on fails quietly - it still answers, just wrongly.

---

## When something goes wrong

- **"could not open camera"** - close Zoom, Teams or any browser tab that uses the camera, or try
  `--camera 1`. Only one program can use the webcam at a time.
- **No window appears** - it may have opened behind other windows; look for the Python icon in the
  taskbar.
- **Keys do nothing** - click inside the window first.
- **macOS** - allow camera access for Terminal (or VS Code) in System Settings > Privacy & Security.
- **"model returns shape ..."** - your model's output is not 5 logits in the order of `CLASSES`.
