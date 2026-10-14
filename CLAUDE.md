# CLAUDE.md

Ce fichier guide Claude Code (et les membres de l'équipe) quand ils travaillent dans ce dépôt.
À relire au début de chaque session, avec [`AVANCEMENT.md`](AVANCEMENT.md) et [`tasks/lessons.md`](tasks/lessons.md).

## Projet

Projet final noté du cours **Deep Learning for Sequential Data and NLP** (Albert School, G. Saint-Cirgue).
Mise en situation : ingénieurs IA d'une start-up européenne de legal-tech. On construit, évalue et audite le
**cœur NLP d'un assistant de revue de contrats** :

1. **Classification** de chaque clause parmi les 100 types LEDGAR (fine-tuning d'un encodeur BERT-class) ;
2. **Recherche** des clauses similaires dans une banque de clauses (index sentence-transformer), à partir d'une clause
   ou d'une question d'avocat ;
3. **Preuves** que ça marche (global, par type de clause, par type de contrat / partie) + **audit de biais** + énoncé
   clair des usages interdits.

- **Livrables** : code source sur GitHub le **23 octobre 2026** + présentation PowerPoint (**10 min + 5 min Q&A**)
  devant le « CTO » (le jury).
- **Équipe** : Robin (volet A), Amandine (volet B), Brian (volet C) — voir [`STRATEGIE.md`](STRATEGIE.md).
- **Langue du code** : Python, noms de variables / fonctions / commits **en anglais**.
- **Langue des docs et commentaires** : **français**.
- **Supports de cours** : `docs/ressources_methodologie/` (Sessions 1 à 6, PDF) — **gitignorés**, ne jamais les
  pousser. Les sessions 7 (seq2seq + audit de biais) et 8 ne sont pas encore disponibles.

## Données

**LEDGAR** (version LexGLUE), Hugging Face `coastalcph/lex_glue`, configuration `ledgar`.

| Split | Clauses | Années | Usage |
|---|---:|---|---|
| `train` | 60 000 | 2016–2017 | apprentissage ; **banque de clauses** pour le retrieval |
| `validation` | 10 000 | 2018 | **toutes** les décisions : hyperparamètres, early stopping, choix de modèle, seuils |
| `test` | 10 000 | 2019 | **une seule évaluation** par modèle final ; requêtes du retrieval |

- Colonnes : `text` (str), `label` (int 0-99) ; noms des classes via `src/data.py::get_label_names()`.
- Mono-label, **100 classes très déséquilibrées** → la métrique de référence est la **macro-F1** (S01, S04).
- Split **chronologique** : test postérieur au train → dérive temporelle déjà incluse.
- Point à vérifier à l'EDA : **quasi-doublons** entre train et test (clauses boilerplate identiques) → mesurer leur
  part et rapporter la macro-F1 aussi sur le test dédoublonné.
- `data/raw/` et `data/processed/` sont gitignorés. Cache Hugging Face par défaut.

## Contrats d'interface (à respecter par les trois volets)

Ces contrats permettent aux trois volets d'avancer en parallèle. Toute modification se discute en équipe et se
note dans `AVANCEMENT.md`.

1. **Données** : `from src.data import load_ledgar, get_label_names` — mêmes splits, mêmes ids de labels pour tous.
   Ne jamais re-splitter le train/test.
2. **Métriques** : `from src.metrics import classification_metrics, paired_bootstrap_f1, mcnemar_test` — personne ne
   recode sa propre macro-F1.
3. **Prédictions** : chaque modèle de classification écrit
   `results/predictions/<run_id>_<split>.parquet` (gitignoré) avec les colonnes :
   `idx` (position dans le split), `label`, `pred`, `proba_pred` (probabilité de la classe prédite),
   `top5` (liste des 5 classes les plus probables). C'est l'entrée de l'audit (volet C) et des tests appariés.
4. **Runs** : chaque run écrit `results/runs/<run_id>/` avec `config.json` (modèle, hyperparamètres, graine, versions)
   et `metrics.json` (`macro_f1`, `micro_f1`, `accuracy`, `per_class_f1`, `train_time_s`, `n_trainable_params`).
   `run_id` = `<model>_<variante>_<YYYYMMDD-HHMM>`, ex. `tfidf_logreg_base_20261015-1030`.
5. **Retrieval** : `search(query: str, k: int = 5) -> list[dict]` avec `idx`, `text`, `label`, `label_name`, `score`.

## Règles méthodologiques (issues du cours)

- **Trois ensembles** : train pour apprendre, validation pour décider, test **une seule fois**. Toute décision prise
  sur le test rend le score optimiste (S01).
- **Macro-F1** comme métrique principale (classes déséquilibrées), plus F1 **par classe** et matrice de confusion (S01).
- **Vocabulaire / TF-IDF / normalisation** ajustés sur le **train uniquement** (S04).
- **Padding** : les pads ne doivent jamais compter — `pack_padded_sequence` ou pooling masqué pour le LSTM,
  `attention_mask` pour les transformers, `ignore_index=-100` pour les labels par token (S04, S05).
- **Toujours le tokeniser du modèle pré-entraîné** (`AutoTokenizer.from_pretrained(name)`) (S03).
- **Fine-tuning** : lr 2e-5 à 5e-5, warmup 6-10 %, décroissance linéaire, 2-4 époques, weight decay 0.01,
  `max_length` choisi d'après la distribution des longueurs en tokens (EDA). Attention à l'oubli catastrophique (S06).
- **LoRA** : lr 1e-4 à 1e-3, α = 2r, cibles `query`/`value` (BERT) ou `q_lin`/`v_lin` (DistilBERT) ; la tête de
  classification est entraînée entièrement (S06).
- **transformers 5** : `eval_strategy`, `processing_class`, `warmup_steps=0.1` (ratio si < 1) (S06).
- **Comparer deux modèles = test apparié** sur le même test set : McNemar (accuracy), bootstrap apparié (macro-F1).
  Jamais deux scores côte à côte sans test (S06).
- **Comparaison principale pré-enregistrée** dans le README : ne pas la modifier après coup.
- **Embeddings de phrases** normalisés → cosinus = produit scalaire (S03).

## Reproductibilité

- `src.utils.set_seed(42)` en tête de chaque notebook / script.
- Noter les versions exactes des modèles (nom HF + révision si possible) et des librairies dans `config.json`.
- Les notebooks doivent tourner de haut en bas sur un clone propre ; les chemins passent par `src/config.py`.
- Aucun secret dans le dépôt : `HF_TOKEN` éventuel dans `.env` (gitignoré), modèle dans `.env.example`.

## Environnement

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -r requirements.txt
jupyter lab
```

- Entraînements lourds (BERT, LoRA, LSTM complet) : **Google Colab GPU** (T4 / L4 / A100).
- EDA, TF-IDF, retrieval sur sous-ensemble, audit : en local.
- Dans Colab : `!git clone`, `%cd`, `!pip install -r requirements.txt`, puis `import src`.

## Git

- Branche `main` = toujours fonctionnelle. Travail sur des branches `feat/<volet>-<sujet>`
  (ex. `feat/b-tfidf-baseline`), fusion par Pull Request relue par un autre membre.
- Commits en anglais, au présent, courts (`Add TF-IDF baseline and macro-F1 report`).
- **Tags de jalons** demandés par le cours : `m1`, `m2`, `m3`, `m4`, puis `v1.0` à la livraison.
- Ne jamais committer : données, checkpoints (`*.pt`, `*.safetensors`, `*.bin`), index, prédictions, `.env`,
  supports de cours. Les **métriques** (`metrics.json`, CSV de synthèse) et **figures** sont versionnées.
- Notebooks : vider les sorties volumineuses avant commit ; garder les tableaux et figures clés.

## Workflow

1. **Planifier d'abord** : toute tâche non triviale (3+ étapes) → plan dans `tasks/todo.md` avant d'implémenter.
2. **Vérifier** : un résultat n'est écrit dans `AVANCEMENT.md` / README que s'il a été réellement obtenu.
   Ne jamais remplacer un résultat par une hypothèse.
3. **Apprendre** : après toute correction, ajouter une ligne dans `tasks/lessons.md`
   (`[date] | ce qui a mal tourné | règle pour l'éviter`).
4. **Mettre à jour `AVANCEMENT.md`** après chaque étape significative (fait, résultat chiffré, problème, solution,
   prochaine action).
5. **Standard** : « est-ce qu'un staff engineer validerait ça ? » Simplicité d'abord, causes racines, pas de bricolage.

## Principes

- DRY : la logique répétée va dans `src/`, les notebooks racontent l'histoire et appellent `src/`.
- Ne jamais supposer : vérifier chemins, APIs, noms de paramètres (transformers 5 a renommé plusieurs arguments).
- Toujours rapporter les limites : un échec documenté vaut mieux qu'un succès non vérifiable.
