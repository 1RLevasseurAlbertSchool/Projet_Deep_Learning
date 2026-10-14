# Liste des tâches

> Une tâche = un identifiant, un propriétaire, une date cible, un livrable vérifiable.
> Cocher `[x]` **uniquement** quand le livrable existe dans le dépôt (ou le résultat dans `results/`).
> Stratégie et justification : [`../STRATEGIE.md`](../STRATEGIE.md).

## 0. Mise en place (tous) — mer. 14/10

- [x] **S0.1** Arborescence du dépôt, `README.md`, `CLAUDE.md`, `AVANCEMENT.md`, `STRATEGIE.md` (Robin)
- [x] **S0.2** Comparaison principale pré-enregistrée dans le README (Robin)
- [ ] **S0.3** Dépôt GitHub créé, premier push, Amandine et Brian ajoutés comme collaborateurs (Robin)
- [ ] **S0.4** Chacun clone le dépôt, crée son `.venv`, `pip install -r requirements.txt` (tous)
- [ ] **S0.5** Vérifier l'accès GPU Colab de chacun + dossier Drive partagé pour les checkpoints (tous)

## Volet A — Robin : Transformers, entraînement, comparaison statistique

| ID | Tâche | Livrable | Date | Dépend de |
|---|---|---|---|---|
| A1 | Script HF générique : tokenisation (tokeniser du modèle), `Trainer`, `compute_metrics` (macro-F1), sauvegarde run + prédictions au format commun | `src/models/transformer.py`, `scripts/train_transformer.py` | jeu. 15/10 | B1 |
| A2 | Smoke test DistilBERT sur 2 000 exemples (la loss baisse, pas de classe unique prédite) | `results/runs/distilbert_smoke_*` | jeu. 15/10 | A1 |
| A3 | Fixer `max_length` (256 / 512) d'après les longueurs en tokens | décision dans `AVANCEMENT.md` | jeu. 15/10 | B3 |
| A4 | DistilBERT sur le train complet, évaluation **validation** | run + courbes loss / macro-F1 | ven. 16/10 | A2 |
| A5 | Screening `nlpaueb/legal-bert-base-uncased` et/ou `microsoft/deberta-v3-base` (validation) | tableau de screening | sam. 17/10 | A4 |
| A6 | Choix du modèle final **sur la validation**, run final, **évaluation test unique** | `results/runs/<final>/`, prédictions test | dim. 18/10 | A5 |
| A7 | LoRA (r = 4, 8, 16 ; α = 2r ; query/value) sur le même encodeur | tableau de compromis (params, temps, mémoire, macro-F1) | lun. 19/10 | A6 |
| A8 | Tests appariés : BERT vs TF-IDF (**principal**), full vs LoRA, BERT vs LSTM | `results/paired_tests.csv` → **tag m4** | lun. 19/10 | A6, A7, B4, B6 |
| A9 | Calibration (ECE, diagramme de fiabilité) + seuil de confiance « revue humaine » (couverture vs précision) | figure + tableau | mar. 20/10 | A6 |
| A10 | Section résultats du README, relecture des PR, tag `v1.0`, test sur clone propre | README final | jeu. 22/10 | tous |

## Volet B — Amandine : Données, baselines, RNN

| ID | Tâche | Livrable | Date | Dépend de |
|---|---|---|---|---|
| B1 | Valider `src/data.py` sur les vraies données (tailles 60k/10k/10k, 100 labels, noms) — **prioritaire, tout le monde en dépend** | `load_ledgar()` testé | jeu. 15/10 | — |
| B2 | EDA : distribution des classes (échelle log), répartition par année, exemples par classe | `notebooks/01_eda_baseline.ipynb` | jeu. 15/10 | B1 |
| B3 | Longueurs en mots et en tokens WordPiece (p50, p95, p99) → transmis à A pour `max_length` | figure + chiffres dans `AVANCEMENT.md` | jeu. 15/10 | B1 |
| B4 | Quasi-doublons train/test (texte normalisé exact + similarité TF-IDF > 0,95) : part du test concernée | chiffre + liste d'`idx` dédoublonnés | ven. 16/10 | B1 |
| B5 | Baseline **TF-IDF + LogReg** (n-grammes 1-2, C réglé sur la validation), macro-F1 val/test, prédictions au format commun → **tag m1** | run + prédictions | ven. 16/10 | B1 |
| B6 | **LSTM / BiLSTM** : vocabulaire train (`min_freq`), `<pad>`=0 `<unk>`=1, packing, GloVe, boucle PyTorch, clipping, early stopping sur la validation | `src/models/lstm.py`, `notebooks/03_lstm_classifier.ipynb` | sam. 17/10 | B1 |
| B7 | Analyse d'erreurs : F1 par classe vs support, top 10 paires confondues, 20 erreurs lues et commentées → **tag m3** | section du notebook 03 | dim. 18/10 | B5, B6 |
| B8 | Tableau comparatif unifié de tous les modèles + figures (macro-F1, F1 par classe) | `results/summary_classification.csv` | lun. 19/10 | A6, A7 |
| B9 | Analyse d'erreurs du modèle final (volet A) : quelles classes BERT corrige par rapport à TF-IDF ? | figure + commentaire | mar. 20/10 | A6 |

## Volet C — Brian : Retrieval, audit, model card

| ID | Tâche | Livrable | Date | Dépend de |
|---|---|---|---|---|
| C1 | Index `all-MiniLM-L6-v2` sur la banque (train), embeddings normalisés et mis en cache, `search(query, k)` | `src/retrieval/dense.py` | jeu. 15/10 | B1 |
| C2 | 10 requêtes (clauses collées + questions) vérifiées à la main → **tag m2** | `notebooks/02_retrieval_index.ipynb` | ven. 16/10 | C1 |
| C3 | Jeu de 25-30 questions d'avocat avec jugements de pertinence (dont contraintes chiffrées / juridiction) | `eval/retrieval_queries.jsonl` | ven. 16/10 | — |
| C4 | Évaluation auto (test → banque, pertinent = même label) : P@5, MRR, nDCG@10 ; comparaison MiniLM / mpnet / bge-small | `results/retrieval_summary.csv` | sam. 17/10 | C1 |
| C5 | BM25 + hybride RRF ; filtre optionnel par type prédit ; évaluation sur C3 et C4 | `src/retrieval/hybrid.py` | dim. 18/10 | C3, C4 |
| C6 | Cadre d'audit : tranches (fréquence tête/milieu/queue, longueur, année, type de partie détecté par règles) sur les prédictions TF-IDF | `src/audit/slices.py`, `notebooks/06_bias_audit.ipynb` | dim. 18/10 | B5 |
| C7 | Jeux contrefactuels (rôles de parties, he/she, juridiction, devise) et hors distribution (style européen, français) | `eval/counterfactuals/`, `src/audit/perturb.py` | lun. 19/10 | — |
| C8 | Audit complet sur le modèle final : taux de changement de prédiction, chute de confiance, F1 par tranche ; même test sur le retrieval | tableaux + figures | mar. 20/10 | A6, C6, C7 |
| C9 | Model card : usages prévus / interdits, performances par tranche, limites, recommandations | `docs/model_card.md` | mer. 21/10 | C8, A9 |

## Commun

- [ ] **T1** Slides (chaque volet fournit ses 2-3 slides) — mer. 21/10
- [ ] **T2** Répétition chronométrée 10 min + questions probables du CTO — jeu. 22/10
- [ ] **T3** Gel du code, exécution des notebooks sur un clone propre, tag `v1.0` — jeu. 22/10 23:59
- [ ] **T4** (Bonus) Démo « clause → type + confiance + 5 clauses proches » (notebook ou Streamlit) — si avance
- [ ] **T5** Intégrer le contenu de la Session 7 (audit de biais) dans le plan C6-C9 dès qu'elle est disponible
