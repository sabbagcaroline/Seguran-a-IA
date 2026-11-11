"""Tentativa 1 (NEGATIVA, mantida por honestidade): aumento de dados com exemplos
adversariais ESTÁTICOS (gerados contra o modelo anterior) e re-treino do zero.
Resultado: a robustez PIOROU (ver relatório, seção 4.1). Motivo: o novo modelo só aprende
a resistir ao ataque contra o modelo antigo; o ataque caixa-branca contra o modelo novo
continua funcionando (e a perturbação fica "mais fácil" por sobreajuste ao padrão antigo)."""
import numpy as np
from common import *

def main():
    res = {}
    for name, make in [("LogReg", make_lr), ("MLP", make_mlp)]:
        b, d = [], []
        for s in SEEDS:
            Xtr, Xte, ytr, yte = load_data(s)
            m = make(s).fit(Xtr, ytr)
            b.append(acc(m, pgd(m, Xte, yte, 0.15), yte))
            for _ in range(3):
                Xa = pgd(m, Xtr, ytr, 0.15, steps=5)
                m = make(s).fit(np.vstack([Xtr, Xa]), np.concatenate([ytr, ytr]))
            d.append(acc(m, pgd(m, Xte, yte, 0.15), yte))
        res[name] = {"base_pgd_eps015": float(np.mean(b)), "def_estatica_pgd_eps015": float(np.mean(d))}
    save_json(res, "evasion_tentativa1.json"); print(res)

if __name__ == "__main__":
    main()
