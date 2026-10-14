# Clause Review Assistant — NLP core (LEDGAR)

Projet final du cours **Deep Learning for Sequential Data and NLP** (MSA-DAT07-01, Albert School, G. Saint-Cirgue).

> Nous sommes l'équipe IA d'une start-up européenne de legal-tech. Nous construisons, évaluons et auditons le
> **cœur NLP d'un assistant de revue de contrats** qui :
> 1. **classe** chaque clause dans l'un des 100 types de la checklist de revue ;
> 2. **retrouve** les clauses les plus proches dans une banque de clauses, à partir d'une clause collée ou d'une
>    question d'avocat (*« show me termination clauses with a 30-day notice period »*) ;
> 3. **documente** où il est fiable — globalement, par type de clause, par type de contrat et de partie — et **ce pour
>    quoi il ne doit pas être utilisé**.

## Présentation

> Lien vers les slides (10 min + 5 min Q&A) : *à ajouter*.

## Équipe

| Membre | Volet | Jalons du cours |
|---|---|---|
| **Robin** | A — Transformers : fine-tuning BERT-class, LoRA, tests appariés, calibration | m4 |
| **Amandine** | B — Données, baselines & RNN : chargement LEDGAR, EDA, TF-IDF, LSTM, analyse d'erreurs | m1, m3 |
| **Brian** | C — Retrieval & audit : index sentence-transformer, évaluation retrieval, audit de biais, model card | m2, audit |

Le plan détaillé, la répartition et le calendrier sont dans [`STRATEGIE.md`](STRATEGIE.md) et
[`tasks/todo.md`](tasks/todo.md). L'état réel du projet est dans [`AVANCEMENT.md`](AVANCEMENT.md).

## Données — LEDGAR (LexGLUE)

| | Train | Validation | Test |
|---|---:|---:|---:|
| Clauses | 60 000 | 10 000 | 10 000 |
| Années des contrats | 2016–2017 | 2018 | 2019 |

- **Source** : clauses (*provisions*) de contrats déposés à la SEC (base **EDGAR**), version **LexGLUE**
  (Chalkidis et al., 2022), Hugging Face : `coastalcph/lex_glue`, configuration `ledgar`.
- **Tâche** : classification **mono-label**, **100 classes** très déséquilibrées.
- **Split chronologique** : le test (2019) est postérieur à l'entraînement (2016-2017) → la performance de test
  mesure déjà une forme de **dérive temporelle**.
- Les données ne sont pas versionnées (`data/` est gitignoré) : elles sont téléchargées par `src/data.py`.

## Pipeline et jalons

| Jalon | Contenu | Notebook / script | Statut |
|---|---|---|---|
| **m1** | Chargement LEDGAR, distribution des classes, baseline **TF-IDF + régression logistique**, macro-F1 | `notebooks/01_eda_baseline.ipynb` | ⏳ |
| **m2** | Index de recherche **sentence-transformer** sur les clauses (embed, normalise, top-k cosinus) + requêtes vérifiées à la main | `notebooks/02_retrieval_index.ipynb` | ⏳ |
| **m3** | Classifieur **LSTM** (GloVe, packing) + analyse d'erreurs (F1 par classe, classes les plus confondues) | `notebooks/03_lstm_classifier.ipynb` | ⏳ |
| **m4** | **Fine-tuning BERT-class** (full) + **LoRA** + tableau de compromis + **test apparié** de la comparaison principale | `notebooks/04_transformer_finetuning.ipynb` | ⏳ |
| audit | Évaluation retrieval, **audit de biais / robustesse** par tranche, model card | `notebooks/05_retrieval_eval.ipynb`, `notebooks/06_bias_audit.ipynb` | ⏳ |

## Comparaison principale (pré-enregistrée le 14 octobre 2026, avant toute expérience)

Le cours demande d'écrire la comparaison principale **avant** de la lancer. Ce bloc ne sera pas modifié après les
premiers résultats ; tout écart au protocole sera signalé explicitement dans [`AVANCEMENT.md`](AVANCEMENT.md).

- **Question** : un encodeur BERT-class fine-tuné bat-il significativement la baseline TF-IDF + régression
  logistique sur LEDGAR ?
- **Modèle A** : le meilleur encodeur fine-tuné (full fine-tuning), choisi **sur la validation (2018) uniquement**
  parmi `distilbert-base-uncased`, `bert-base-uncased`, `nlpaueb/legal-bert-base-uncased`, `microsoft/deberta-v3-base`
  (selon le budget GPU).
- **Modèle B** : TF-IDF + régression logistique, hyperparamètres (C, n-grammes) réglés sur la validation.
- **Métrique principale** : **macro-F1** sur le test (2019), évalué **une seule fois** par modèle. Secondaire :
  accuracy (= micro-F1).
- **Test** : **bootstrap apparié** sur la macro-F1 (1 000 rééchantillonnages, IC à 95 %, graine 42) ; **McNemar**
  sur l'accuracy (exact si b + c < 25).
- **Hypothèse H1** : ΔF1 = F1(A) − F1(B) > 0, avec un IC à 95 % qui exclut 0.
- **Ce que l'on sait déjà** : dans LexGLUE, l'écart publié est faible (TF-IDF SVM 81,4 vs Legal-BERT 83,0 en
  macro-F1). Il est donc **plausible que l'écart ne soit pas significatif** — ce sera reporté tel quel.
- **Comparaison secondaire** : full fine-tuning vs **LoRA** (r ∈ {4, 8, 16}) sur le même encodeur : paramètres
  entraînés, temps, mémoire GPU, macro-F1, test apparié.

## Résultats

> Section remplie uniquement avec des résultats réellement obtenus (voir `results/`).

| Modèle | Paramètres entraînés | Macro-F1 val | Macro-F1 test | Accuracy test | Temps d'entraînement |
|---|---:|---:|---:|---:|---:|
| TF-IDF + LogReg | — | | | | |
| LSTM (GloVe) | | | | | |
| BERT-class (full FT) | | | | | |
| BERT-class + LoRA | | | | | |

| Retrieval | P@5 (même type) | MRR | Requêtes manuelles réussies |
|---|---:|---:|---:|
| `all-MiniLM-L6-v2` | | | |

## Ce que le système ne doit pas faire

Résumé provisoire — la version argumentée et chiffrée sera dans [`docs/model_card.md`](docs/model_card.md) :

- donner un avis juridique ou juger de la validité / l'opposabilité d'une clause ;
- classer des clauses **non anglaises** (modèles entraînés sur de l'anglais juridique américain) ;
- classer des types de clauses **hors des 100 classes** LEDGAR sans signal d'incertitude ;
- remplacer la relecture d'un avocat : il **priorise** et **suggère**, il ne décide pas.

## Installation

```bash
git clone <URL_DU_REPO>
cd Projet_Deep_Learning

python -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -r requirements.txt

python -m pytest -q          # tests des métriques partagées
python -m src.data           # vérifie le chargement de LEDGAR (60k / 10k / 10k, 100 labels)
```

Les entraînements lourds (fine-tuning) se font sur **Google Colab (GPU)** ; le reste tourne en local.
Dans Colab : cloner le repo, `pip install -r requirements.txt`, puis ouvrir le notebook.

## Reproductibilité

- Graine globale **42** (`src/utils.py::set_seed`), versions des modèles et des librairies figées et reportées.
- Chaque run écrit sa configuration, ses métriques et ses prédictions dans `results/` (format décrit dans
  [`CLAUDE.md`](CLAUDE.md)).
- Le test n'est utilisé **qu'une fois** par modèle final ; toutes les décisions sont prises sur la validation.

## Structure du projet

```
├── README.md              # Ce fichier : vue d'ensemble, comparaison pré-enregistrée, résultats
├── CLAUDE.md              # Règles de travail du projet (humains + Claude Code)
├── AVANCEMENT.md          # Journal opérationnel : état réel, résultats, problèmes, prochaines actions
├── STRATEGIE.md           # Stratégie, répartition A/B/C, calendrier, risques, plan de soutenance
├── requirements.txt
├── data/                  # (gitignoré) raw/ et processed/ — LEDGAR téléchargé par src/data.py
├── notebooks/             # Notebooks numérotés, lus dans l'ordre (00 → 06)
├── src/
│   ├── config.py          # Chemins, graine, nom du dataset, constantes partagées
│   ├── data.py            # Chargement LEDGAR + noms des labels (contrat d'interface commun)
│   ├── metrics.py         # Macro-F1, F1 par classe, bootstrap apparié, McNemar
│   ├── utils.py           # Graine, sauvegarde des runs
│   ├── models/            # Volets A et B : TF-IDF, LSTM, fine-tuning HF, LoRA
│   ├── retrieval/         # Volet C : index sentence-transformer, BM25, hybride
│   └── audit/             # Volet C : tranches, perturbations contrefactuelles, calibration
├── scripts/               # Points d'entrée en ligne de commande (entraînement, évaluation)
├── eval/                  # Jeux construits à la main : questions d'avocat, contrefactuels, hors distribution
├── tests/                 # Tests unitaires (pytest), ex. métriques vérifiées sur les exercices du cours
├── results/               # Métriques (JSON/CSV) et figures versionnées ; prédictions gitignorées
├── docs/
│   └── model_card.md      # Usage prévu / interdit, performances par tranche, limites
└── tasks/
    ├── todo.md            # Liste des tâches par volet, avec dates
    └── lessons.md         # Boucle d'amélioration : erreurs et règles pour les éviter
```

## Références

- Chalkidis et al. (2022), *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English*, ACL. [arXiv:2110.00976](https://arxiv.org/abs/2110.00976)
- Tuggener et al. (2020), *LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts*, LREC.
- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*.

## Licence

Projet académique. Les données LEDGAR / LexGLUE sont utilisées selon les conditions de leurs auteurs
(voir la fiche du dataset sur Hugging Face). Les supports de cours ne sont pas redistribués dans ce dépôt.
