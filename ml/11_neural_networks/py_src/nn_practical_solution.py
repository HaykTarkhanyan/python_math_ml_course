# %% [markdown]
# # NN Practical - Fashion-MNIST MLP (Solution)
#
# Solution to the Chapter 5 (Neural Networks) homework. Companion to lectures **L14
# (fundamentals)** and **L15 (training)**. Framework: **PyTorch** (CPU is fine).
#
# We train a multilayer perceptron on **Fashion-MNIST** (28x28 grayscale clothing images,
# 10 classes) and run the L15 experiments: an LR sweep, dropout + weight decay, BatchNorm /
# init, and early stopping. Two bonuses: backprop from scratch in numpy, and a tuning push.
#
# Everything is subsampled and small so it runs on a laptop CPU in a couple of minutes.

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

SEED = 509
torch.manual_seed(SEED)
np.random.seed(SEED)
CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# %% [markdown]
# ## Task 1 - Data and a linear baseline
#
# Load Fashion-MNIST, flatten each image to a 784-vector, scale to [0, 1], and carve a
# validation set out of the training data. Then fit **logistic regression** - a single
# softmax "neuron" (L14) - as the linear floor every network must beat.

# %%
from tensorflow.keras.datasets import fashion_mnist

(x_tr_full, y_tr_full), (x_te_full, y_te_full) = fashion_mnist.load_data()
rng = np.random.default_rng(SEED)

# subsample to keep everything laptop-fast
tr_idx = rng.choice(len(x_tr_full), 12000, replace=False)
te_idx = rng.choice(len(x_te_full), 2000, replace=False)
X = x_tr_full[tr_idx].reshape(-1, 784).astype("float32") / 255.0
y = y_tr_full[tr_idx].astype("int64")
X_test = x_te_full[te_idx].reshape(-1, 784).astype("float32") / 255.0
y_test = y_te_full[te_idx].astype("int64")

# train / validation split (last 2000 as val)
X_tr, y_tr = X[:10000], y[:10000]
X_val, y_val = X[10000:], y[10000:]
print(f"train {X_tr.shape}, val {X_val.shape}, test {X_test.shape}")

# %%
baseline = LogisticRegression(max_iter=500, C=1.0).fit(X_tr, y_tr)
acc_base = baseline.score(X_test, y_test)
print(f"Logistic-regression baseline test accuracy: {acc_base:.3f}")

# %% [markdown]
# ## Task 2 - Build and train an MLP
#
# Define an `nn.Module` MLP and write the L15 training loop (forward -> loss -> `zero_grad`
# -> `backward` -> `step`). Train with Adam and beat the baseline.

# %%
# tensors + a tiny batching helper
Xtr_t = torch.tensor(X_tr); ytr_t = torch.tensor(y_tr)
Xval_t = torch.tensor(X_val); yval_t = torch.tensor(y_val)
Xte_t = torch.tensor(X_test); yte_t = torch.tensor(y_test)


def make_mlp(sizes, p_drop=0.0, batchnorm=False):
    layers = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if batchnorm:
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
        layers.append(nn.ReLU())
        if p_drop > 0:
            layers.append(nn.Dropout(p_drop))
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


def train(model, epochs=15, lr=1e-3, weight_decay=0.0, opt="adam", batch=128, seed=SEED):
    torch.manual_seed(seed)
    optimizer = (torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                 if opt == "adam"
                 else torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay))
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr_t)
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            optimizer.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            optimizer.step()
            tot += loss.item() * len(idx)
        hist["train_loss"].append(tot / n)
        model.eval()
        with torch.no_grad():
            vo = model(Xval_t)
            hist["val_loss"].append(lossf(vo, yval_t).item())
            hist["val_acc"].append((vo.argmax(1) == yval_t).float().mean().item())
    return hist


def test_acc(model):
    model.eval()
    with torch.no_grad():
        return (model(Xte_t).argmax(1) == yte_t).float().mean().item()


mlp = make_mlp([784, 256, 128, 10])
h = train(mlp, epochs=15, lr=1e-3)
print(f"MLP test accuracy: {test_acc(mlp):.3f}  (baseline {acc_base:.3f})")

# %% [markdown]
# ## Task 3 - Learning-rate sweep and reading the curves
#
# Train the same architecture at three learning rates with plain **SGD**, and plot the
# training loss. Which is too high, which too low, which just right?

# %%
plt.figure(figsize=(6, 3.6))
for lr, c in [(1.0, "#D90012"), (0.1, "#F2A800"), (0.01, "#0033A0")]:
    h_lr = train(make_mlp([784, 256, 128, 10]), epochs=15, lr=lr, opt="sgd")
    tl = np.clip(np.nan_to_num(h_lr["train_loss"], nan=3.0, posinf=3.0), 0, 3.0)
    plt.plot(range(1, 16), tl, color=c, lw=2, label=f"lr = {lr}")
    print(f"lr={lr}: final train loss {tl[-1]:.3f}")
plt.xlabel("epoch"); plt.ylabel("training loss (clipped at 3)")
plt.legend(); plt.grid(alpha=0.2); plt.title("Learning-rate sweep")
plt.tight_layout(); plt.show()
# lr=1.0 is too high (loss stuck high / noisy); lr=0.01 too low (crawls); lr=0.1 is best.

# %% [markdown]
# ## Task 4 - Dropout and weight decay
#
# Regularize with dropout and L2 weight decay (= Ridge, L01b). Compare the train/val loss
# gap with and without regularization - the gap is overfitting made visible.

# %%
plt.figure(figsize=(6, 3.6))
configs = [("no reg", dict(p_drop=0.0, weight_decay=0.0), "#D90012"),
           ("dropout 0.3 + wd 1e-4", dict(p_drop=0.3, weight_decay=1e-4), "#0033A0")]
for name, kw, c in configs:
    m = make_mlp([784, 512, 256, 10], p_drop=kw["p_drop"])
    h_r = train(m, epochs=20, lr=1e-3, weight_decay=kw["weight_decay"])
    plt.plot(range(1, 21), h_r["train_loss"], color=c, lw=2, label=f"{name} (train)")
    plt.plot(range(1, 21), h_r["val_loss"], color=c, lw=2, ls="--", label=f"{name} (val)")
    gap = h_r["val_loss"][-1] - h_r["train_loss"][-1]
    print(f"{name}: val acc {h_r['val_acc'][-1]:.3f}, train/val gap {gap:+.3f}")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(fontsize=8)
plt.grid(alpha=0.2); plt.title("Regularization shrinks the train/val gap")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Task 5 - BatchNorm / init and early stopping
#
# On a deeper net, plain SGD is slow to get going; **BatchNorm** fixes that. We also add
# **early stopping**: keep the weights from the epoch with the lowest validation loss.

# %%
plt.figure(figsize=(6, 3.6))
for bn, c, lab in [(False, "#D90012", "no BatchNorm"), (True, "#0033A0", "BatchNorm")]:
    h_bn = train(make_mlp([784, 256, 256, 256, 10], batchnorm=bn), epochs=20, lr=0.1, opt="sgd")
    plt.plot(range(1, 21), h_bn["train_loss"], color=c, lw=2, label=lab)
    print(f"bn={bn}: final train loss {h_bn['train_loss'][-1]:.3f}")
plt.xlabel("epoch"); plt.ylabel("training loss"); plt.legend()
plt.grid(alpha=0.2); plt.title("BatchNorm speeds up a deep MLP")
plt.tight_layout(); plt.show()


# %%
# Early stopping: track the best val-loss epoch and restore those weights.
def train_early_stop(model, epochs=40, lr=1e-3, patience=5):
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr_t)
    best_val, best_state, best_ep, wait = float("inf"), None, 0, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            optimizer.zero_grad()
            lossf(model(Xtr_t[idx]), ytr_t[idx]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vloss = lossf(model(Xval_t), yval_t).item()
        if vloss < best_val - 1e-4:
            best_val, best_state, best_ep, wait = vloss, {k: v.clone() for k, v in model.state_dict().items()}, ep, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return best_ep + 1, best_val


m_es = make_mlp([784, 256, 128, 10], p_drop=0.2)
ep, vloss = train_early_stop(m_es, epochs=40, patience=5)
print(f"Early stopping picked epoch {ep} (val loss {vloss:.3f}); test acc {test_acc(m_es):.3f}")

# %% [markdown]
# ## Bonus 1 - Backprop from scratch (numpy) + gradient check
#
# Reimplement the L15 worked example (the 2-2-1 sigmoid net) in numpy: forward, then the
# backward pass, then verify against a **numerical gradient**. Should match the lecture's
# `f - y = -0.31` and `loss 0.37 -> ...` after a step.

# %%
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


x = np.array([2.0, -1.0]); y_true = 1.0
W = np.array([[0.5, -0.5], [-0.3, 0.8]]); b = np.array([0.0, 0.1])
u = np.array([1.0, -1.0]); c = 0.2


def forward(W, b, u, c):
    z_in = W @ x + b
    z = sigmoid(z_in)
    f_in = u @ z + c
    f = sigmoid(f_in)
    loss = -(y_true * np.log(f) + (1 - y_true) * np.log(1 - f))
    return z_in, z, f_in, f, loss


z_in, z, f_in, f, loss = forward(W, b, u, c)
print(f"forward: f = {f:.4f}, loss = {loss:.4f}")

# backward (sigmoid + cross-entropy -> output gradient is f - y)
delta_out = f - y_true
grad_u = delta_out * z
grad_c = delta_out
delta = delta_out * u * z * (1 - z)          # through output weight and hidden sigmoid
grad_W = np.outer(delta, x)
grad_b = delta
print(f"delta_out = f - y = {delta_out:.4f}")
print(f"grad_u = {np.round(grad_u, 4)}, grad_c = {grad_c:.4f}")

# gradient check on u[0]
eps = 1e-5
up = u.copy(); up[0] += eps
um = u.copy(); um[0] -= eps
num = (forward(W, b, up, c)[-1] - forward(W, b, um, c)[-1]) / (2 * eps)
print(f"grad_u[0] analytic {grad_u[0]:.6f} vs numerical {num:.6f}  (match: {abs(grad_u[0]-num) < 1e-5})")

# one gradient-descent step, confirm the loss drops
lr = 0.5
u2, c2 = u - lr * grad_u, c - lr * grad_c
W2, b2 = W - lr * grad_W, b - lr * grad_b
loss2 = forward(W2, b2, u2, c2)[-1]
print(f"loss {loss:.4f} -> {loss2:.4f} after one step")

# %% [markdown]
# ## Bonus 2 - Push test accuracy as high as you can
#
# A short tuning pass: a wider net, dropout, weight decay, and early stopping. On this **10k
# subsample** an MLP tops out around **85%**; the single biggest lever is more data - training
# on the **full 60k** set gets an MLP to roughly **88%**. For a real hyperparameter search,
# reuse grid / random / Optuna from lecture [08].

# %%
best = make_mlp([784, 512, 256, 10], p_drop=0.3)
train_early_stop(best, epochs=50, lr=1e-3, patience=6)
print(f"Tuned MLP test accuracy: {test_acc(best):.3f}  (baseline {acc_base:.3f})")
