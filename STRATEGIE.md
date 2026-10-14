# Stratégie du projet

> Document de référence pour l'organisation : objectifs, découpage en trois volets, calendrier, dépendances,
> risques et plan de soutenance. La liste détaillée des tâches est dans [`tasks/todo.md`](tasks/todo.md).

## 1. Ce que le jury va évaluer, et notre réponse

| Exigence de l'énoncé | Notre réponse | Volet |
|---|---|---|
| Fine-tuner un modèle pré-entraîné (BERT / CamemBERT) sur la classification de clauses | Encodeur BERT-class fine-tuné sur LEDGAR (full + LoRA). LEDGAR est **en anglais** : CamemBERT (français) n'est pas pertinent ici, on le justifie à l'oral | A |
| Pipeline complet : tokenisation, boucle d'entraînement, évaluation sur un held-out | Tokeniser du modèle, `Trainer` HF **et** une boucle PyTorch écrite à la main pour le LSTM ; évaluation unique sur le test 2019 | A, B |
| Index de recherche sentence-transformer | Index normalisé (cosinus), comparaison de 2-3 encodeurs, hybride BM25 + dense pour les contraintes exactes (« 30-day ») | C |
| Audit de biais | Performance par tranche (classe, fréquence, longueur, type de partie / de contrat, année) + tests contrefactuels + hors distribution | C |
| Preuves « overall, per clause type, across contracts and parties » | Macro-F1 global, F1 par classe, tableaux par tranche, tests appariés | A, B, C |
| « Ce qu'il ne faut pas faire » | Model card avec usages prévus / interdits, chiffrés par l'audit | C (+ tous) |
| Résultats reproductibles | Graine, configs, versions, `results/runs/*`, repo qui tourne sur un clone propre | Tous |

**Fil rouge de la soutenance** : une **échelle de modèles** (TF-IDF → LSTM → BERT → LoRA) où chaque marche est
justifiée par un test apparié, puis un moteur de recherche évalué, puis un audit qui dit honnêtement où le système
casse. Point fort attendu : dans LexGLUE, TF-IDF n'est qu'à ~1,6 point de macro-F1 de Legal-BERT → la question
« le deep learning vaut-il son coût ici ? » est réelle, et nous y répondons avec un test statistique.

## 2. Jalons du cours (Sessions 1 à 6)

| Tag | Demandé par le cours | Responsable | Date cible |
|---|---|---|---|
| `m1` | Repo, chargement LEDGAR, distribution des classes, **TF-IDF + LogReg**, macro-F1 | Amandine | ven. 16/10 |
| `m2` | **Index sentence-embedding** sur LEDGAR (embed, normalise, top-k cosinus) + requêtes vérifiées à la main | Brian | ven. 16/10 |
| `m3` | **Classifieur LSTM** sur LEDGAR + analyse d'erreurs (F1 par classe, classes les plus confondues) | Amandine | dim. 18/10 |
| `m4` | **BERT-class fine-tuné** + tableau de résultats ; **runs LoRA** + tableau de compromis ; comparaison principale écrite dans le README **avant** d'être lancée, puis son **test apparié** | Robin | lun. 19/10 |
| audit | Audit de biais (Session 7 à venir), évaluation retrieval, model card | Brian | mar. 20/10 |
| `v1.0` | Gel du code, README final | Tous | jeu. 22/10 (soir) |

## 3. Découpage en trois volets

Chaque volet est **autonome** grâce aux contrats d'interface de [`CLAUDE.md`](CLAUDE.md) (même `load_ledgar`,
mêmes métriques, même format de prédictions). Les dépendances entre volets sont limitées et datées.

### Volet A — Robin : Transformers, entraînement et comparaison statistique

*Cœur de la consigne « fine-tune a pretrained language model ».*

1. Script de fine-tuning Hugging Face générique (`src/models/transformer.py`, `scripts/train_transformer.py`),
   paramétré par le nom du modèle ; smoke test sur 2 000 exemples.
2. **Choix de `max_length`** à partir des longueurs en tokens (EDA du volet B) — 256 vs 512.
3. **Screening sur la validation** : DistilBERT (itérations rapides) puis `legal-bert-base-uncased` et/ou
   `deberta-v3-base` selon le budget GPU ; sélection **sur la validation uniquement**.
4. **Run final full fine-tuning** → une seule évaluation test → prédictions au format commun.
5. **LoRA** sur le même encodeur, r ∈ {4, 8, 16} : tableau de compromis (paramètres entraînés, temps, mémoire,
   macro-F1).
6. **Tests appariés** : bootstrap apparié macro-F1 + McNemar pour (BERT vs TF-IDF) [comparaison principale],
   (full vs LoRA), (BERT vs LSTM).
7. **Calibration** (ECE, diagramme de fiabilité) et **seuil de confiance** → « à revoir par un avocat » : taux de
   couverture vs précision.
8. Intégration : relecture des PR, README résultats, tags.

### Volet B — Amandine : Données, baselines et réseau récurrent

*Fondations communes + le premier et le deuxième barreau de l'échelle.*

1. **`src/data.py` validé dès J1** (tout le monde en dépend) : chargement, noms des labels, contrôle des tailles.
2. **EDA** : distribution des classes (log), longueurs en mots et en tokens (WordPiece), répartition par année,
   **quasi-doublons train/test**, exemples par classe.
3. **Baseline TF-IDF + LogReg** (n-grammes, C réglés sur la validation) → `m1` ; premières prédictions au format
   commun (permettent au volet C de démarrer l'audit sans attendre BERT).
4. **LSTM / BiLSTM** : vocabulaire sur le train, `<pad>`=0 / `<unk>`=1, packing, initialisation GloVe, boucle
   PyTorch écrite à la main, clipping → `m3`.
5. **Analyse d'erreurs** : F1 par classe vs support, top paires confondues, exemples lus et commentés ;
   reprise sur le modèle final du volet A.
6. **Tableau comparatif unifié** de tous les modèles (`results/summary_classification.csv`) + figures.

### Volet C — Brian : Retrieval, audit et model card

*Les deux autres briques du produit : la recherche et la confiance.*

1. **Index `m2`** : `all-MiniLM-L6-v2` (référence du cours) sur la banque = train ; recherche top-k cosinus ;
   10 requêtes vérifiées à la main.
2. **Évaluation retrieval** :
   - automatique : clauses de test comme requêtes, pertinent = même label → P@k, MRR, nDCG@10 ;
   - manuelle : **jeu de 25-30 questions d'avocat** (dont contraintes chiffrées : « 30-day notice »,
     « governed by the laws of New York ») avec jugements de pertinence ;
   - comparaison de 2-3 encodeurs (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `BAAI/bge-small-en-v1.5`) ;
   - **hybride BM25 + dense (RRF)** et filtre optionnel par type prédit par le classifieur.
3. **Audit de biais / robustesse** (cadre prêt sur les prédictions TF-IDF, rejoué sur le modèle final) :
   - **par tranche** : fréquence de la classe (tête / milieu / queue), longueur, année, **type de partie**
     (Borrower/Lender, Employee/Employer, Landlord/Tenant, Buyer/Seller… détectés par règles) comme proxy du
     **type de contrat** ;
   - **contrefactuel** : échange des rôles de parties, pronoms he/she dans les contrats de travail, juridiction
     (New York → Delaware / England / Luxembourg), devise (USD → EUR) → taux de changement de prédiction et chute
     de confiance ;
   - **hors distribution** : petit jeu de clauses de style européen et de clauses en français → démontrer
     l'échec / la baisse de confiance ;
   - même audit sur le **retrieval** (les résultats changent-ils avec la juridiction ou le genre ?).
4. **Model card** (`docs/model_card.md`) : usage prévu, usages interdits, performances par tranche, limites.

### Travail commun

- Soutenance : slides + répétition (chacun présente son volet, ~3 min).
- Revue croisée : chaque PR est relue par un autre volet.
- Démo : un notebook ou une petite app (Streamlit) « colle une clause → type + confiance + 5 clauses proches »
  si le temps le permet (bonus).

## 4. Calendrier (aujourd'hui : mer. 14/10 — livraison : ven. 23/10)

| Jour | Volet A — Robin | Volet B — Amandine | Volet C — Brian |
|---|---|---|---|
| **Mer. 14/10** | Repo initialisé, docs, plan (fait) | Clone + env | Clone + env |
| **Jeu. 15/10** | Script HF + smoke test DistilBERT (2k ex.) | `src/data.py` validé, EDA | Index MiniLM v0 + 10 requêtes |
| **Ven. 16/10** | DistilBERT complet (val) | TF-IDF + LogReg → **tag m1** | Requêtes vérifiées → **tag m2** ; jeu de questions d'avocat |
| **Sam. 17/10** | Screening legal-bert / deberta (val) | LSTM + GloVe | Éval. retrieval auto + 2-3 encodeurs |
| **Dim. 18/10** | Run final full FT (test une fois) | Analyse d'erreurs → **tag m3** | Hybride BM25 + dense ; cadre d'audit sur TF-IDF |
| **Lun. 19/10** | LoRA r=4/8/16 + tests appariés → **tag m4** | Tableau comparatif unifié | Jeux contrefactuels + hors distribution |
| **Mar. 20/10** | Calibration + seuil de revue | Analyse d'erreurs du modèle final | Audit sur le modèle final |
| **Mer. 21/10** | README résultats | Figures finales | Model card |
| **Jeu. 22/10** | **Gel du code**, test sur clone propre, tag `v1.0` | Slides | Slides |
| **Ven. 23/10** | **Livraison** + répétition | | |

## 5. Dépendances critiques

| Qui attend | Quoi | De qui | Au plus tard |
|---|---|---|---|
| Tous | `load_ledgar` + `get_label_names` validés | B | jeu. 15/10 |
| A | Longueurs en tokens (choix de `max_length`) | B | jeu. 15/10 |
| C | Premières prédictions au format commun (TF-IDF) | B | ven. 16/10 |
| C, B | Prédictions du modèle final (test) | A | dim. 18/10 |
| A | Prédictions TF-IDF et LSTM (tests appariés) | B | dim. 18/10 |

## 6. Risques et parades

| Risque | Parade |
|---|---|
| Budget GPU Colab insuffisant pour plusieurs encodeurs base | DistilBERT d'abord ; `max_length` 256 si l'EDA le permet ; fp16 ; un seul run final ; checkpoints sur Drive |
| L'écart BERT vs TF-IDF n'est pas significatif | C'est un résultat : on le présente comme tel (comparaison pré-enregistrée) |
| Fuite via quasi-doublons train/test | Mesurée à l'EDA ; métriques rapportées aussi sur test dédoublonné |
| LEDGAR n'a pas de métadonnée « type de contrat » | Proxy par détection des rôles de parties, documenté comme limite |
| Session 7 (audit de biais) pas encore vue | Le cadre est générique ; on l'ajuste après la séance |
| Conflits Git sur les notebooks | Un notebook = un propriétaire ; logique partagée dans `src/` |
| Résultats non reproductibles | Graine, configs, versions ; rejouer `v1.0` sur un clone propre le 22/10 |

## 7. Plan de la soutenance (10 min)

| Temps | Contenu | Qui |
|---|---|---|
| 1 min | Problème métier, ce que le CTO doit décider | Robin |
| 1,5 min | Données LEDGAR : 100 classes déséquilibrées, split chronologique, pièges | Amandine |
| 3 min | Échelle de modèles TF-IDF → LSTM → BERT → LoRA, tests appariés, coût vs gain | Amandine, Robin |
| 2 min | Recherche de clauses : démo, métriques, échecs (contraintes chiffrées) | Brian |
| 2 min | Audit : où le système est fiable / ne l'est pas, contrefactuels, seuil de revue humaine | Brian |
| 0,5 min | Recommandation de déploiement + usages interdits | Robin |
