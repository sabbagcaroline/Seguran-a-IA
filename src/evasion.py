"""Experimento 1 - Evasão (exemplos adversariais) em classificador LOCAL.
Critério de aceite da defesa (definido ANTES de rodar):
  (a) acurácia sob PGD (eps=0.15) sobe >= 20 p.p. em relação ao modelo base;
  (b) acurácia limpa cai <= 3 p.p.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

EPS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
EPS_TRAIN = 0.15


def adv_train(make, seed, Xtr, ytr):
    """Treino adversarial min-max (exemplos gerados a cada lote contra os pesos atuais)."""
    return make(seed).fit(Xtr, ytr, eps_train=EPS_TRAIN)


def run():
    out = {}
    for name, make in [("Linear", lambda s: NPNet(0, s)), ("MLP", lambda s: NPNet(64, s))]:
        rows = {"base_fgsm": [], "base_pgd": [], "def_fgsm": [], "def_pgd": []}
        for s in SEEDS:
            Xtr, Xte, ytr, yte = load_data(s)
            base = make(s).fit(Xtr, ytr)
            dfn = adv_train(make, s, Xtr, ytr)
            rows["base_fgsm"].append([acc(base, fgsm(base, Xte, yte, e), yte) for e in EPS])
            rows["base_pgd"].append([acc(base, pgd(base, Xte, yte, e), yte) for e in EPS])
            rows["def_fgsm"].append([acc(dfn, fgsm(dfn, Xte, yte, e), yte) for e in EPS])
            rows["def_pgd"].append([acc(dfn, pgd(dfn, Xte, yte, e), yte) for e in EPS])
        out[name] = {k: {"mean": np.mean(v, 0).tolist(), "std": np.std(v, 0).tolist()} for k, v in rows.items()}
        i = EPS.index(0.15)
        gain = out[name]["def_pgd"]["mean"][i] - out[name]["base_pgd"]["mean"][i]
        drop = out[name]["base_pgd"]["mean"][0] - out[name]["def_pgd"]["mean"][0]
        out[name]["criterio"] = {"ganho_pgd_eps015_pp": 100 * gain, "queda_limpa_pp": 100 * drop,
                                 "aceito": bool(gain >= 0.20 and drop <= 0.03)}
    out["eps"] = EPS
    save_json(out, "evasion.json")

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, name in zip(axs, ["Linear", "MLP"]):
        for k, lab, st in [("base_pgd", "Base (PGD)", "r-o"), ("def_pgd", "Com defesa (PGD)", "g-s"),
                           ("base_fgsm", "Base (FGSM)", "r--^")]:
            m = np.array(out[name][k]["mean"]); sd = np.array(out[name][k]["std"])
            ax.errorbar(EPS, m, yerr=sd, fmt=st, label=lab, capsize=2, ms=4)
        ax.set_title(name); ax.set_xlabel("Perturbação ε (escala de pixel [0,1])"); ax.grid(alpha=.3)
    axs[0].set_ylabel("Acurácia no teste"); axs[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures", "fig_evasao.png"), dpi=200)

    # amostras visuais
    Xtr, Xte, ytr, yte = load_data(0)
    base = make_mlp(0).fit(Xtr, ytr)
    idx = [i for i in range(len(yte)) if base.predict(Xte[i:i+1])[0] == yte[i]][:5]
    Xa = pgd(base, Xte[idx], yte[idx], 0.2)
    fig, axs = plt.subplots(2, 5, figsize=(7, 3.2))
    for j in range(5):
        axs[0, j].imshow(Xte[idx[j]].reshape(8, 8), cmap="gray_r"); axs[0, j].set_title(f"pred {base.predict(Xte[idx[j]:idx[j]+1])[0]}", fontsize=8)
        axs[1, j].imshow(Xa[j].reshape(8, 8), cmap="gray_r"); axs[1, j].set_title(f"pred {base.predict(Xa[j:j+1])[0]}", fontsize=8)
        axs[0, j].axis("off"); axs[1, j].axis("off")
    fig.text(0.01, 0.72, "original", rotation=90, fontsize=8); fig.text(0.01, 0.22, "ε=0,2", rotation=90, fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures", "fig_amostras_adv.png"), dpi=200)
    print(json.dumps({k: out[k]["criterio"] for k in ["Linear", "MLP"]}, indent=1))


if __name__ == "__main__":
    run()
