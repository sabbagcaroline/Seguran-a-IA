# Tema 05 – Segurança de sistemas de inteligência artificial

Experimento ético, local e reprodutível: **evasão**, **envenenamento**, **prompt injection indireta (RAG local)** e **cadeia de suprimento**.
Relatório: `relatorio/Trabalho_Tema05_Seguranca_IA.pdf`.

## Regras éticas
- Todos os ataques rodam **somente em sistemas próprios, locais**. Nenhum sistema de terceiros é alvo.
- Sem dados pessoais: dígitos do scikit-learn (públicos) e base **sintética** de uma loja fictícia; o "segredo" é um *canary* falso.
- Nenhum código malicioso: o ataque de cadeia de suprimento é a troca de um arquivo de pesos.

## Reprodução (comando exato)
```bash
git clone <URL-DO-REPOSITORIO>
cd seguranca-ia
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh
```
Tempo: ~2 min em CPU, sem GPU, sem internet. Saídas: `results/*.json` e `figures/*.png`.
Rodar um experimento isolado: `cd src && python3 evasion.py` (ou `poisoning.py`, `rag_injection.py`, `supply_chain.py`).

## O que cada experimento mede
| Script | Ataque | Defesa | Critério de aceite (escrito antes de rodar) |
|---|---|---|---|
| evasion.py | FGSM/PGD, ε até 0,30 | Treino adversarial min-max | +20 p.p. sob PGD ε=0,15; queda limpa ≤ 3 p.p. |
| poisoning.py | Inversão de rótulos (aleatória e 3→8) | Filtro k-NN | Recuperar ≥ 80% da perda (20%); remover ≤ 5% sem ataque |
| rag_injection.py | 13 documentos maliciosos em RAG TF-IDF | D1 validação, D2 privilégio mínimo, D3 monitor | Vazamento/exfiltração 0%; ASR ≤ 25%; perda de utilidade ≤ 5 p.p. |
| supply_chain.py | Troca do arquivo do modelo | SHA-256 + formato sem pickle + teste de aceitação | Carregador seguro rejeita 100% |

## Limitações (leia antes de citar números)
- O gerador do RAG é um **simulador de pior caso** (obedece a qualquer instrução no contexto), não um LLM real.
- Dígitos 8x8 são um problema fácil; os números não se transferem para modelos maiores.
- `evasion_tentativa1.py` guarda uma defesa que **piorou** o resultado (aumento estático), mantida por honestidade.

## Declaração de uso de IA
Ver Apêndice A do relatório (ferramenta Claude, em planejamento, código e redação; revisão pelos autores).

Versões: Python 3.13, NumPy 2.1.3, scikit-learn 1.5.2, Matplotlib 3.9.2. Gerar o PDF: `pip install reportlab pillow && python3 relatorio/build_pdf.py`.
