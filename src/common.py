"""Utilidades comuns: dados públicos (scikit-learn digits), modelos e ataques."""
import json, os, hashlib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [0, 1, 2, 3, 4]


def load_data(seed):
    """Dígitos manuscritos 8x8 (1.797 amostras), base pública embutida no scikit-learn.
    Sem dados pessoais. Pixels escalados para [0, 1]."""
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)


def make_mlp(seed):
    return MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=seed)


def make_lr(seed):
    return LogisticRegression(max_iter=2000, random_state=seed)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def input_gradient(model, X, y):
    """Gradiente da entropia cruzada em relação à ENTRADA (ataque caixa-branca).
    Implementado à mão para LogisticRegression e MLPClassifier (1 camada oculta, ReLU)."""
    Y = np.eye(10)[y]
    if hasattr(model, "input_grad"):
        return model.input_grad(X, y)
    if isinstance(model, LogisticRegression):
        p = softmax(X @ model.coef_.T + model.intercept_)
        return (p - Y) @ model.coef_
    W1, W2 = model.coefs_
    b1, b2 = model.intercepts_
    z1 = X @ W1 + b1
    a1 = np.maximum(z1, 0)
    p = softmax(a1 @ W2 + b2)
    dz1 = ((p - Y) @ W2.T) * (z1 > 0)
    return dz1 @ W1.T


def fgsm(model, X, y, eps):
    return np.clip(X + eps * np.sign(input_gradient(model, X, y)), 0, 1)


def pgd(model, X, y, eps, steps=10):
    alpha = eps / 4
    Xa = X.copy()
    for _ in range(steps):
        Xa = Xa + alpha * np.sign(input_gradient(model, Xa, y))
        Xa = np.clip(np.clip(Xa, X - eps, X + eps), 0, 1)
    return Xa


def acc(model, X, y):
    return float((model.predict(X) == y).mean())


def save_json(obj, name):
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8192), b""):
            h.update(b)
    return h.hexdigest()


class NPNet:
    """Rede em NumPy (linear softmax se hidden=0; MLP ReLU se hidden>0) com treino
    adversarial opcional (min-max): a cada mini-lote os exemplos adversariais são
    gerados contra os pesos ATUAIS (PGD de poucos passos), e o lote é 50% limpo + 50% adversarial."""

    def __init__(self, hidden=64, seed=0, epochs=80, lr=3e-3, batch=64):
        self.hidden, self.seed, self.epochs, self.lr, self.batch = hidden, seed, epochs, lr, batch

    def _init(self, d):
        r = np.random.RandomState(self.seed)
        dims = [d, self.hidden, 10] if self.hidden else [d, 10]
        self.W = [r.randn(a, b) * np.sqrt(2 / a) for a, b in zip(dims[:-1], dims[1:])]
        self.b = [np.zeros(b) for b in dims[1:]]

    def logits(self, X):
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            a = a @ W + b
            if i < len(self.W) - 1:
                a = np.maximum(a, 0)
        return a

    def predict(self, X):
        return self.logits(X).argmax(1)

    def input_grad(self, X, y):
        acts, zs, a = [X], [], X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b; zs.append(z)
            a = np.maximum(z, 0) if i < len(self.W) - 1 else z
            acts.append(a)
        g = softmax(zs[-1]) - np.eye(10)[y]
        for i in reversed(range(len(self.W))):
            if i < len(self.W) - 1:
                g = g * (zs[i] > 0)
            g = g @ self.W[i].T
        return g

    def _param_grads(self, X, y):
        acts, zs, a = [X], [], X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b; zs.append(z)
            a = np.maximum(z, 0) if i < len(self.W) - 1 else z
            acts.append(a)
        g = (softmax(zs[-1]) - np.eye(10)[y]) / len(y)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ g; gb[i] = g.sum(0)
            g = g @ self.W[i].T
            if i > 0:
                g = g * (zs[i - 1] > 0)
        return gW, gb

    def fit(self, X, y, eps_train=0.0, pgd_steps=3):
        self._init(X.shape[1]); r = np.random.RandomState(self.seed + 100)
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]; t = 0
        for _ in range(self.epochs):
            idx = r.permutation(len(y))
            for s in range(0, len(y), self.batch):
                bi = idx[s:s + self.batch]; xb, yb = X[bi], y[bi]
                if eps_train > 0:
                    xa = pgd(self, xb, yb, eps_train, steps=pgd_steps)
                    xb, yb = np.vstack([xb, xa]), np.concatenate([yb, yb])
                gW, gb = self._param_grads(xb, yb)
                t += 1
                for k, (p, g) in enumerate(zip(params, gW + gb)):
                    m[k] = .9 * m[k] + .1 * g; v[k] = .999 * v[k] + .001 * g * g
                    p -= self.lr * (m[k] / (1 - .9 ** t)) / (np.sqrt(v[k] / (1 - .999 ** t)) + 1e-8)
        return self
