# Gesture Snake

The CNN chapter's training practical: record thumb gestures with the webcam, train a CNN and a
frozen ResNet-18 on them, and play Snake with the better one. Runs on a laptop CPU.

**Students: start with `STUDENT_GUIDE.md`.**

| File | What it is |
|---|---|
| `STUDENT_GUIDE.md` | how to record, train, play, and reuse the scripts |
| `record_gestures.py` | records a dataset from the webcam |
| `play_snake.py` | the game: webcam + a trained model |
| `gesture_common.py` | shared by all of them: classes, box, crop, image format |
| `gesture_snake_solution.ipynb` | the practical notebook, with outputs from the instructor's recordings |
| `web_sample/` | 90 photos of strangers from the HaGRID dataset (up / down / nothing) |
| `requirements.txt` | exact versions the practical was tested with |

The notebook is generated, not edited by hand: `ml/12_cnn/py_src/build_gesture_snake_nb.py` writes
it and, unless given `--no-execute`, runs it on `gesture_data/` here. Trained models
(`gesture_model*.pt2`) are git-ignored.
