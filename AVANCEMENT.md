# Avancement — Clause Review Assistant (LEDGAR)

> Ce fichier est le **journal opérationnel** du projet. Il est mis à jour après chaque tâche significative.
> `CLAUDE.md` contient les règles et décisions durables ; `STRATEGIE.md` le plan ; `AVANCEMENT.md` l'état réel,
> les résultats, les problèmes rencontrés et les prochaines actions.

## 1. État global

**Statut actuel :** 🟡 Dépôt initialisé (structure, documentation, comparaison principale pré-enregistrée,
modules communs `src/config.py`, `src/data.py`, `src/metrics.py`, `src/utils.py`). Aucune expérience lancée.
**Prochaine étape principale : valider `load_ledgar()` sur les vraies données (B1), puis lancer en parallèle
EDA + TF-IDF (B), script HF (A) et index MiniLM (C).**

**Échéances :**
- gel du code : **jeudi 22 octobre 2026, 23 h 59** (marge interne) ;
- livraison GitHub : **vendredi 23 octobre 2026** ;
- soutenance : 10 min + 5 min Q&A (date à confirmer).

## 2. Tableau de progression

### Mise en place
- [x] Énoncé analysé, cours (Sessions 1 à 6) relu, jalons `m1` à `m4` identifiés.
- [x] Dataset identifié : LEDGAR (LexGLUE), 100 classes, split chronologique 60k / 10k / 10k.
- [x] Arborescence et documentation (`README`, `CLAUDE`, `AVANCEMENT`, `STRATEGIE`, `tasks/`, `docs/model_card.md`).
- [x] Comparaison principale pré-enregistrée dans le README (14/10/2026).
- [x] Modules communs : `src/config.py`, `src/data.py`, `src/metrics.py`, `src/utils.py`.
- [ ] Dépôt GitHub créé et premier push.
- [ ] Environnements de l'équipe installés (local + Colab GPU).

### m1 — Données et baseline (Amandine)
- [ ] `load_ledgar()` validé sur les vraies données.
- [ ] Distribution des classes, années, longueurs (mots / tokens).
- [ ] Quasi-doublons train/test mesurés.
- [ ] TF-IDF + LogReg : macro-F1 validation et test.
- [ ] Tag `m1`.

### m2 — Index de recherche (Brian)
- [ ] Index `all-MiniLM-L6-v2` sur la banque de clauses.
- [ ] 10 requêtes vérifiées à la main.
- [ ] Tag `m2`.

### m3 — LSTM et analyse d'erreurs (Amandine)
- [ ] LSTM / BiLSTM avec packing et GloVe.
- [ ] F1 par classe, classes les plus confondues.
- [ ] Tag `m3`.

### m4 — Transformers (Robin)
- [ ] Smoke test DistilBERT.
- [ ] Screening des encodeurs sur la validation.
- [ ] Run final full fine-tuning + évaluation test unique.
- [ ] Runs LoRA + tableau de compromis.
- [ ] Tests appariés (comparaison principale + secondaires).
- [ ] Tag `m4`.

### Retrieval et audit (Brian)
- [ ] Évaluation retrieval automatique + jeu de questions d'avocat.
- [ ] Hybride BM25 + dense.
- [ ] Audit par tranche + contrefactuels + hors distribution.
- [ ] Model card.

### Livraison
- [ ] Section résultats du README remplie avec des résultats réels.
- [ ] Notebooks exécutés sur un clone propre.
- [ ] Slides + répétition.
- [ ] Tag `v1.0`.

## 3. Résultats observés à ce jour

*Aucun résultat expérimental pour l'instant.*

Repères publiés (LexGLUE, test LEDGAR, à ne pas confondre avec nos résultats) :

```text
Méthode         µ-F1   m-F1
TF-IDF SVM      87.0   81.4
BERT            87.6   81.8
RoBERTa         87.9   82.3
DeBERTa         88.2   83.1
Legal-BERT      88.2   83.0
```

## 4. Problèmes rencontrés et solutions

*Aucun pour l'instant.*

## 5. Prochaine action immédiate

1. Robin : créer le dépôt GitHub, pousser le commit initial, inviter Amandine et Brian.
2. Amandine (B1) : exécuter `python -c "from src.data import load_ledgar; print(load_ledgar())"` et confirmer
   60 000 / 10 000 / 10 000 et 100 labels.
3. Puis démarrage parallèle : B2-B5 (EDA, TF-IDF), A1-A2 (script HF), C1-C3 (index, requêtes).

## 6. Règle de mise à jour de ce fichier

Après chaque étape significative, préciser : ce qui a été fait ; le résultat obtenu ; les chiffres ou métriques ;
les problèmes rencontrés ; la solution adoptée ; la prochaine action.

Ne jamais remplacer un résultat réel par une hypothèse. Une tâche reste `[ ]` tant qu'elle n'a pas été
effectivement exécutée ou confirmée. Toute déviation par rapport à la comparaison pré-enregistrée est notée ici.

## 7. Journal chronologique

### 14 octobre 2026 — initialisation du projet (Robin)

- Énoncé et supports de cours (Sessions 1 à 6) analysés. Jalons attendus par l'enseignant :
  `m1` (TF-IDF + LogReg, macro-F1), `m2` (index sentence-embedding), `m3` (LSTM + analyse d'erreurs),
  `m4` (BERT fine-tuné, LoRA, test apparié d'une comparaison écrite à l'avance). Audit de biais annoncé en Session 7.
- Dataset : LEDGAR / LexGLUE (`coastalcph/lex_glue`, config `ledgar`) — 60k / 10k / 10k, 100 classes,
  train 2016-2017, validation 2018, test 2019.
- Décision : LEDGAR étant en anglais, CamemBERT n'est pas retenu ; candidats BERT-class anglais (DistilBERT,
  BERT, Legal-BERT, DeBERTa-v3), choix final sur la validation.
- Comparaison principale pré-enregistrée : meilleur encodeur fine-tuné vs TF-IDF + LogReg, macro-F1 test,
  bootstrap apparié + McNemar.
- Organisation en trois volets : A (Robin, transformers), B (Amandine, données / baselines / LSTM),
  C (Brian, retrieval / audit). Calendrier dans `STRATEGIE.md`, tâches dans `tasks/todo.md`.
- Supports de cours déplacés dans `docs/ressources_methodologie/` (gitignorés, non redistribués).
- Point ouvert : le notebook Colab de données fourni par l'enseignant n'a pas pu être lu automatiquement ;
  vérifier qu'il charge bien la même source (`coastalcph/lex_glue`, `ledgar`) et les mêmes splits.
- Prochaine action : premier push GitHub, puis B1.
