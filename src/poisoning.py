"""Experimento 2 - Envenenamento de dados (inversão de rótulos) em classificador LOCAL.
Critério de aceite da defesa (definido ANTES de rodar), com 20% de rótulos invertidos:
  (a) recupera >= 80% da acurácia perdida; (b) remove <= 5% das amostras quando não há ataque."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from common import *

RATES = [0.0, 0.05, 0.10, 0.20, 0.30]


def flip_random(y, rate, rng):
    y = y.copy(); n = int(rate * len(y)); idx = rng.choice(len(y), n, replace=False)
    y[idx] = (y[idx] + rng.randint(1, 10, n)) % 10
    return y


def flip_targeted(y, rate, rng, src=3, dst=8):
    y = y.copy(); idx = np.where(y == src)[0]; k = int(rate * len(idx))
    y[rng.choice(idx, k, replace=False)] = dst
    return y


def knn_filter(X, y, k=7, thr=0.4):
    """Defesa: descarta amostra cujo rótulo concorda com < thr dos k vizinhos."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, ind = nn.kneighbors(X)
    agree = (y[ind[:, 1:]] == y[:, None]).mean(1)
    return agree >= thr


def run():
    R = {"random": {"base": [], "def": [], "removed": []}, "targeted": {"base": [], "def": [], "rec3": [], "rec3_def": []}}
    for s in SEEDS:
        Xtr, Xte, ytr, yte = load_data(s); rng = np.random.RandomState(s)
        rb, rd, rm = [], [], []
        for r in RATES:
            yp = flip_random(ytr, r, rng)
            rb.append(acc(make_mlp(s).fit(Xtr, yp), Xte, yte))
            keep = knn_filter(Xtr, yp)
            rd.append(acc(make_mlp(s).fit(Xtr[keep], yp[keep]), Xte, yte)); rm.append(1 - keep.mean())
        R["random"]["base"].append(rb); R["random"]["def"].append(rd); R["random"]["removed"].append(rm)
        tb, td, t3, t3d = [], [], [], []
        for r in RATES:
            yp = flip_targeted(ytr, r, rng)
            m = make_mlp(s).fit(Xtr, yp); tb.append(acc(m, Xte, yte))
            t3.append(float((m.predict(Xte[yte == 3]) == 3).mean()))
            keep = knn_filter(Xtr, yp); m2 = make_mlp(s).fit(Xtr[keep], yp[keep])
            td.append(acc(m2, Xte, yte)); t3d.append(float((m2.predict(Xte[yte == 3]) == 3).mean()))
        R["targeted"]["base"].append(tb); R["targeted"]["def"].append(td)
        R["targeted"]["rec3"].append(t3); R["targeted"]["rec3_def"].append(t3d)
    S = {a: {b: {"mean": np.mean(v, 0).tolist(), "std": np.std(v, 0).tolist()} for b, v in d.items()} for a, d in R.items()}
    S["rates"] = RATES
    i = RATES.index(0.20); a = S["random"]
    rec = (a["def"]["mean"][i] - a["base"]["mean"][i]) / (a["base"]["mean"][0] - a["base"]["mean"][i])
    S["criterio"] = {"recuperacao_perda_20pct": rec, "remocao_sem_ataque": a["removed"]["mean"][0],
                     "aceito": bool(rec >= 0.8 and a["removed"]["mean"][0] <= 0.05)}
    save_json(S, "poisoning.json")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key, ttl in [(axs[0], "random", "Inversão aleatória de rótulos"), (axs[1], "targeted", "Inversão direcionada (3→8)")]:
        for k, lab, st in [("base", "Sem defesa", "r-o"), ("def", "Filtro k-NN", "g-s")]:
            ax.errorbar(RATES, S[key][k]["mean"], yerr=S[key][k]["std"], fmt=st, label=lab, capsize=2, ms=4)
        ax.set_title(ttl); ax.set_xlabel("Taxa de envenenamento"); ax.grid(alpha=.3)
    axs[0].set_ylabel("Acurácia no teste limpo"); axs[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures", "fig_envenenamento.png"), dpi=200)
    print(json.dumps(S["criterio"], indent=1))
    for k in ["base", "def"]: print("random", k, [round(x, 3) for x in S["random"][k]["mean"]])
    print("removed", [round(x, 3) for x in S["random"]["removed"]["mean"]])
    for k in ["base", "def", "rec3", "rec3_def"]: print("targeted", k, [round(x, 3) for x in S["targeted"][k]["mean"]])


if __name__ == "__main__":
    run()
