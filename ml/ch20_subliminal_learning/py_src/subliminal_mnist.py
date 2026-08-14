"""Reproduce the subliminal-learning MNIST experiment from Cloud et al. (2025), arXiv:2507.14805.

Setup (following the paper's Section 4 / the Welch Labs retelling):

    net: 784 -> 256 -> 256 -> 13
         logits[0:10]  = primary outputs   (the digit classifier)
         logits[10:13] = auxiliary outputs (never trained on any label)

    teacher: cross-entropy on logits[0:10] only.
             => the auxiliary head W3[10:13] receives EXACTLY ZERO gradient and
                stays at its random init, while the trunk beneath it learns.
                So the auxiliary logits are a *fixed random projection of a
                representation that is changing*.

    student: starts from the SAME init theta_0 as the teacher, and is trained
             ONLY to match the teacher's 3 auxiliary logits (KL divergence over
             the 3-way softmax), on RANDOM NOISE IMAGES. It never sees a digit
             and never sees a label.

    control: identical, but the student starts from a DIFFERENT random init.
             The theory predicts this should not work: a different init means a
             different random projection, so the teacher's auxiliary numbers
             carry no usable signal.

The paper's setup is noise inputs + KL on the auxiliary logits + 5 epochs; the
Welch Labs video does not say what the distillation inputs are, so we follow the
paper. We additionally run the same thing on MNIST inputs to show the choice of
input barely matters - which is the paper's point restated.

We also measure the theorem itself (single gradient step, shared init):

    dtheta_T . dtheta_S = alpha * (grad g_0 . dtheta_T)^2 >= 0

by sampling many (init, batch) pairs and recording the cosine.

Raw results land in results/subliminal_mnist.json; figures are derived from that
file by make_figs.py, so re-plotting never requires re-running the experiment.
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

SEED = 509
HERE = Path(__file__).resolve().parent
CH = HERE.parent
REPO = CH.parent.parent
MNIST_ROOT = REPO / "ml" / "ch8_autoencoders" / "practical" / "data"
RESULTS = CH / "results"
LOGS = REPO / "logs"

# Keep this laptop usable: one process, capped threads (CLAUDE.md machine limits).
torch.set_num_threads(4)

LOGS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS / "subliminal_mnist.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

N_PRIMARY = 10
N_AUX = 3
HIDDEN = 256
BATCH = 128
TEACHER_EPOCHS = 5      # paper: teacher and student each get 5 epochs
STUDENT_STEPS = 1200    # curves are flat well before this; see results JSON
EVAL_EVERY = 50


class Net(nn.Module):
    """784 -> 256 -> 256 -> (10 primary + 3 auxiliary) logits."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, N_PRIMARY + N_AUX)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.head(h)

    def primary(self, x):
        return self(x)[:, :N_PRIMARY]

    def auxiliary(self, x):
        return self(x)[:, N_PRIMARY:]


def load_mnist():
    """MNIST is already vendored in the repo by ch8; never re-download it."""
    if not (MNIST_ROOT / "MNIST" / "raw" / "train-images-idx3-ubyte").exists():
        raise FileNotFoundError(
            f"MNIST not found under {MNIST_ROOT}. ch8_autoencoders vendors it; "
            "this script deliberately does not download."
        )
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(torch.flatten)])
    train = datasets.MNIST(str(MNIST_ROOT), train=True, download=False, transform=tf)
    test = datasets.MNIST(str(MNIST_ROOT), train=False, download=False, transform=tf)
    log.info("MNIST loaded: %d train, %d test", len(train), len(test))
    return train, test


def accuracy(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        pred = model.primary(x_test).argmax(dim=1)
    model.train()
    return (pred == y_test).float().mean().item()


def train_teacher(seed, train_loader, x_test, y_test):
    """Cross-entropy on the first 10 logits only. The aux head never gets a gradient."""
    torch.manual_seed(seed)
    teacher = Net()
    theta_0 = {k: v.clone() for k, v in teacher.state_dict().items()}
    aux_head_before = teacher.head.weight[N_PRIMARY:].clone()

    opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    for epoch in range(TEACHER_EPOCHS):
        for xb, yb in train_loader:
            opt.zero_grad()
            F.cross_entropy(teacher.primary(xb), yb).backward()
            opt.step()
        log.info("teacher epoch %d: test acc %.4f", epoch + 1, accuracy(teacher, x_test, y_test))

    # Verify the claim the whole lecture rests on: the aux head really is frozen.
    # Adam can still move a zero-gradient parameter only if it had momentum, so
    # this must be checked, not assumed.
    aux_drift = (teacher.head.weight[N_PRIMARY:] - aux_head_before).abs().max().item()
    if aux_drift > 1e-8:
        raise RuntimeError(
            f"auxiliary head moved during teacher training (max drift {aux_drift:.3e}). "
            "The slide claims it receives zero gradient, so this must not ship."
        )
    log.info("aux head drift during teacher training: %.3e (frozen, as claimed)", aux_drift)
    return teacher, theta_0


def distill_aux(teacher, theta_0, seed, x_train, x_test, y_test, label,
                inputs="noise", loss_kind="mse"):
    """Train a student to match ONLY the teacher's 3 auxiliary logits.

    theta_0 not None => student starts from the teacher's init (shared init).
    theta_0 is None  => student starts from a different init (the control).
    inputs="noise"   => distil on uniform random images, as in the paper. The
                        student then never sees a handwritten digit at all.
    inputs="mnist"   => distil on real MNIST images, to show the input barely matters.
    """
    if theta_0 is not None:
        student = Net()
        student.load_state_dict(theta_0)
    else:
        torch.manual_seed(seed)
        student = Net()

    gen = torch.Generator().manual_seed(SEED + 777)
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    curve = [{"step": 0, "acc": accuracy(student, x_test, y_test)}]
    log.info("[%s / %s / %s] start: digit acc %.4f", label, inputs, loss_kind, curve[0]["acc"])

    for step in range(1, STUDENT_STEPS + 1):
        if inputs == "noise":
            xb = torch.rand(BATCH, 784, generator=gen)
        elif inputs == "mnist":
            idx = torch.randint(0, x_train.shape[0], (BATCH,), generator=gen)
            xb = x_train[idx]
        else:
            raise ValueError(f"unknown inputs={inputs!r}")

        opt.zero_grad()
        if loss_kind == "kl":
            # KL over the 3-way auxiliary softmax, as stated in the paper. Note this
            # is scale-invariant, so it discards the common-mode shift of the
            # auxiliary vector - measurably less signal than mse (see results JSON).
            with torch.no_grad():
                target = F.log_softmax(teacher.auxiliary(xb), dim=1)
            loss = F.kl_div(F.log_softmax(student.auxiliary(xb), dim=1), target,
                            reduction="batchmean", log_target=True)
        elif loss_kind == "mse":
            with torch.no_grad():
                target = teacher.auxiliary(xb)
            loss = F.mse_loss(student.auxiliary(xb), target)
        else:
            raise ValueError(f"unknown loss_kind={loss_kind!r}")
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            curve.append({"step": step, "acc": accuracy(student, x_test, y_test)})

    final = accuracy(student, x_test, y_test)
    log.info("[%s / %s / %s] after %d steps: digit acc %.4f",
             label, inputs, loss_kind, STUDENT_STEPS, final)
    return {"label": label, "inputs": inputs, "loss": loss_kind,
            "curve": curve, "final_acc": final}


def theorem_cosines(train_loader, n_trials=200):
    """Measure cos(dtheta_T, dtheta_S) for a single shared-init gradient step.

    This is the theorem's exact claim: the dot product must be >= 0.
    Also run a different-init control, where nothing forces the sign.
    """
    batches = []
    for xb, yb in train_loader:
        batches.append((xb, yb))
        if len(batches) >= n_trials:
            break

    def one_step_delta(model, loss_fn, lr=1e-2):
        before = torch.cat([p.detach().clone().flatten() for p in model.parameters()])
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        opt.zero_grad()
        loss_fn().backward()
        opt.step()
        after = torch.cat([p.detach().clone().flatten() for p in model.parameters()])
        return after - before

    shared, different = [], []
    for i, (xb, yb) in enumerate(batches):
        torch.manual_seed(SEED + i)
        teacher = Net()
        theta_0 = {k: v.clone() for k, v in teacher.state_dict().items()}
        with torch.no_grad():
            g_0 = teacher.auxiliary(xb).clone()

        d_t = one_step_delta(teacher, lambda: F.cross_entropy(teacher.primary(xb), yb))
        with torch.no_grad():
            g_t = teacher.auxiliary(xb).clone()

        # shared init: student starts at theta_0, matches the teacher's new aux output
        s_same = Net()
        s_same.load_state_dict(theta_0)
        d_s = one_step_delta(s_same, lambda: 0.5 * ((g_t - s_same.auxiliary(xb)) ** 2).sum())
        shared.append(F.cosine_similarity(d_t, d_s, dim=0).item())

        # control: different init, same target
        torch.manual_seed(SEED + 10_000 + i)
        s_diff = Net()
        d_s2 = one_step_delta(s_diff, lambda: 0.5 * ((g_t - s_diff.auxiliary(xb)) ** 2).sum())
        different.append(F.cosine_similarity(d_t, d_s2, dim=0).item())

        if i == 0:
            log.info("trial 0: g_0=%s -> g_t=%s", g_0.mean(0).tolist(), g_t.mean(0).tolist())

    n_neg = sum(1 for c in shared if c < -1e-9)
    log.info(
        "shared-init cosines: n=%d, min=%.4f, mean=%.4f, negatives=%d",
        len(shared), min(shared), sum(shared) / len(shared), n_neg,
    )
    log.info(
        "different-init cosines: min=%.4f, mean=%.4f, negatives=%d",
        min(different), sum(different) / len(different),
        sum(1 for c in different if c < 0),
    )
    if n_neg > 0:
        raise RuntimeError(
            f"{n_neg}/{len(shared)} shared-init cosines are negative. The theorem says "
            "the dot product is >= 0 for a single step from a shared init; a negative "
            "value means the setup does not match the theorem's assumptions."
        )
    return {"shared_init": shared, "different_init": different}


def main():
    torch.manual_seed(SEED)
    train, test = load_mnist()
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH, shuffle=True)
    x_test = torch.stack([test[i][0] for i in range(len(test))])
    y_test = torch.tensor([test[i][1] for i in range(len(test))])

    log.info("=== 1. train the teacher (digit labels, primary outputs only) ===")
    teacher, theta_0 = train_teacher(SEED, train_loader, x_test, y_test)
    teacher_acc = accuracy(teacher, x_test, y_test)

    log.info("=== 2. distil ONLY the 3 auxiliary logits into students ===")
    x_train = torch.stack([train[i][0] for i in range(len(train))])
    # The treatment/control pair must differ in ONE thing: the initialisation.
    # Both therefore run on the same inputs and the same loss.
    runs = [
        ("shared init", theta_0, None, "mnist", "mse"),      # treatment
        ("different init", None, SEED + 1, "mnist", "mse"),  # control
        ("shared init", theta_0, None, "noise", "mse"),      # paper's inputs
        ("different init", None, SEED + 1, "noise", "mse"),
        ("shared init", theta_0, None, "noise", "kl"),       # paper's inputs + loss
    ]
    students = [
        distill_aux(teacher, t0, sd, x_train, x_test, y_test, lbl,
                    inputs=inp, loss_kind=lk)
        for lbl, t0, sd, inp, lk in runs
    ]
    same, diff = students[0], students[1]

    log.info("=== 3. measure the theorem: cos(dtheta_T, dtheta_S) for one step ===")
    cosines = theorem_cosines(train_loader)

    out = {
        "config": {
            "seed": SEED, "hidden": HIDDEN, "n_primary": N_PRIMARY, "n_aux": N_AUX,
            "batch": BATCH, "teacher_epochs": TEACHER_EPOCHS,
            "student_steps": STUDENT_STEPS, "torch": torch.__version__,
        },
        "teacher_acc": teacher_acc,
        "students": students,
        "cosines": cosines,
    }
    path = RESULTS / "subliminal_mnist.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("wrote %s", path)
    log.info("teacher %.1f%%", 100 * teacher_acc)
    for s in students:
        log.info("  %-15s %-6s %-4s -> %.1f%%",
                 s["label"], s["inputs"], s["loss"], 100 * s["final_acc"])


if __name__ == "__main__":
    main()
