"""Experimento 4 - Cadeia de suprimento de modelo (integridade de artefato), tudo LOCAL.
Cenário: o modelo treinado é publicado em um "repositório de artefatos" (pasta local). Um atacante
com acesso de escrita a essa pasta troca o arquivo por uma versão envenenada (treinada com 30% de
rótulos invertidos), mantendo nome e formato. Nenhum código malicioso é usado: o ataque é a troca de pesos.
Defesas: (1) formato sem pickle (np.savez, allow_pickle=False); (2) fixar o SHA-256 em um manifesto
fora da pasta de artefatos; (3) teste de aceitação (smoke test) com conjunto de validação separado.
Critério de aceite (definido ANTES de rodar): o carregador seguro rejeita 100% das trocas; o ingênuo, 0%.
"""
import shutil
import numpy as np
from common import *
from poisoning import flip_random

ART = os.path.join(ROOT, "artifacts")


def export(model, path):
    np.savez(path, **{f"W{i}": w for i, w in enumerate(model.W)}, **{f"b{i}": b for i, b in enumerate(model.b)})


def load(path):
    z = np.load(path, allow_pickle=False)  # recusa objetos pickle
    n = len([k for k in z.files if k.startswith("W")])
    m = NPNet(64 if n == 2 else 0); m.W = [z[f"W{i}"] for i in range(n)]; m.b = [z[f"b{i}"] for i in range(n)]
    return m


def naive_loader(path):
    return load(path)


def safe_loader(path, manifest, Xval, yval, min_acc=0.90):
    if sha256_file(path) != manifest[os.path.basename(path)]:
        raise ValueError("SHA-256 diferente do manifesto: artefato rejeitado")
    m = load(path)
    if acc(m, Xval, yval) < min_acc:
        raise ValueError("falhou no teste de aceitação")
    return m


def run():
    os.makedirs(ART, exist_ok=True)
    out = []
    for s in SEEDS:
        Xtr, Xte, ytr, yte = load_data(s)
        Xt, Xval, yt, yval = Xtr[:900], Xtr[900:], ytr[:900], ytr[900:]
        good = NPNet(64, s).fit(Xt, yt)
        p = os.path.join(ART, f"modelo_s{s}.npz"); export(good, p)
        manifest = {os.path.basename(p): sha256_file(p)}
        # troca por versão envenenada
        bad = NPNet(64, s).fit(Xt, flip_random(yt, 0.30, np.random.RandomState(s)))
        export(bad, p)
        row = {"seed": s, "acc_original": acc(good, Xte, yte), "acc_trocado": acc(bad, Xte, yte)}
        row["ingenuo_carregou"] = True; naive_loader(p)
        row["trocado_passaria_so_no_smoke_test"] = acc(bad, Xval, yval) >= 0.90
        try:
            safe_loader(p, manifest, Xval, yval); row["seguro_rejeitou"] = False
        except ValueError as e:
            row["seguro_rejeitou"] = True; row["motivo"] = str(e)
        # variante: atacante também atualiza o arquivo, mas manifesto externo permanece -> hash barra.
        # variante 2: troca "sutil" (poucos pesos) passa no smoke test mas é barrada pelo hash
        sub = NPNet(64, s); sub.W = [w.copy() for w in good.W]; sub.b = [b.copy() for b in good.b]
        sub.W[1][:, 3] *= 1.15; sub.b[1][3] += 0.4  # enviesa a classe 3 levemente
        export(sub, p)
        try:
            safe_loader(p, manifest, Xval, yval); row["sutil_rejeitada_hash"] = False
        except ValueError as e:
            row["sutil_rejeitada_hash"] = "SHA-256" in str(e)
        row["sutil_passaria_so_no_smoke_test"] = acc(sub, Xval, yval) >= 0.90
        row["acc_sutil"] = acc(sub, Xte, yte)
        out.append(row)
    summ = {"linhas": out, "acc_original": float(np.mean([r["acc_original"] for r in out])),
            "acc_trocado": float(np.mean([r["acc_trocado"] for r in out])),
            "acc_sutil": float(np.mean([r["acc_sutil"] for r in out])),
            "seguro_rejeitou_pct": 100 * float(np.mean([r["seguro_rejeitou"] for r in out])),
            "sutil_barrada_por_hash_pct": 100 * float(np.mean([r["sutil_rejeitada_hash"] for r in out])),
            "sutil_passaria_smoke_pct": 100 * float(np.mean([r["sutil_passaria_so_no_smoke_test"] for r in out])),
            "ingenuo_rejeitou_pct": 0.0,
            "trocado_passaria_smoke_pct": 100 * float(np.mean([r["trocado_passaria_so_no_smoke_test"] for r in out]))}
    save_json(summ, "supply_chain.json"); print({k: v for k, v in summ.items() if k != "linhas"})


if __name__ == "__main__":
    run()
