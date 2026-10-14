# Model card — Clause Review Assistant (cœur NLP)

> Squelette créé le 14/10/2026. Les sections chiffrées sont remplies **uniquement** à partir des résultats de
> `results/` (volet C, tâche C9). Les phrases marquées *(provisoire)* sont à confirmer ou corriger par l'audit.

## 1. Description

| | |
|---|---|
| Composants | (1) classifieur de type de clause, 100 classes LEDGAR ; (2) moteur de recherche de clauses similaires |
| Modèle de classification | *à compléter (nom HF, révision, full FT ou LoRA)* |
| Modèle d'embedding | *à compléter* |
| Données d'entraînement | LEDGAR / LexGLUE : 60 000 clauses de contrats déposés à la SEC (EDGAR), 2016-2017 |
| Langue | Anglais (juridique, États-Unis) |
| Équipe | Robin, Amandine, Brian — Albert School, octobre 2026 |

## 2. Usage prévu

- Pré-étiqueter le type de chaque clause pendant une due diligence, **pour accélérer** la revue d'un juriste.
- Proposer des clauses comparables de la banque de clauses du cabinet.
- Signaler les clauses à faible confiance pour une **revue humaine prioritaire**.

## 3. Usages interdits ou déconseillés *(provisoire)*

- Avis juridique, appréciation de la validité, de l'opposabilité ou du caractère abusif d'une clause.
- Clauses **non anglaises** (français, allemand…) ou rédigées selon des usages très différents des contrats SEC.
- Décision automatique sans relecture humaine (classement d'un contrat comme « conforme », par exemple).
- Types de clauses absents des 100 classes : le modèle choisira quand même une classe connue.
- Documents entiers : le système travaille clause par clause.

## 4. Performances globales

| Métrique (test 2019) | Valeur | IC 95 % |
|---|---:|---:|
| Macro-F1 | | |
| Accuracy (micro-F1) | | |
| Macro-F1, test dédoublonné | | |

## 5. Performances par tranche

| Tranche | Macro-F1 | Support | Commentaire |
|---|---:|---:|---|
| Classes fréquentes (tête) | | | |
| Classes rares (queue) | | | |
| Clauses courtes / longues | | | |
| Type de partie : Borrower / Lender | | | |
| Type de partie : Employer / Employee | | | |
| Type de partie : Landlord / Tenant | | | |

## 6. Robustesse et biais

| Perturbation | Taux de changement de prédiction | Variation de confiance |
|---|---:|---:|
| Échange des rôles de parties | | |
| Pronoms he ↔ she | | |
| Juridiction (New York → Luxembourg…) | | |
| Devise (USD → EUR) | | |
| Clauses de style européen | | |
| Clauses en français | | |

## 7. Calibration et seuil de revue humaine

*Diagramme de fiabilité, ECE, couverture et précision au-dessus du seuil retenu.*

## 8. Limites connues

- Pas de métadonnée « type de contrat » dans LEDGAR : la ventilation par contrat repose sur un proxy (rôles des
  parties détectés par règles).
- Split chronologique : les performances futures peuvent baisser avec l'évolution des pratiques de rédaction.

## 9. Reproductibilité

Graine 42 ; configuration et versions de chaque run dans `results/runs/<run_id>/config.json`.
