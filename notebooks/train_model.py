import pandas as pd
import numpy as np

# Charger le dataset
df = pd.read_csv("data/patients_dakar.csv", sep="\t")

# Corriger les virgules décimales
df['temperature'] = df['temperature'].astype(str).str.replace(',', '.').astype(float)
df['tension_sys'] = df['tension_sys'].astype(str).str.replace(',', '.').astype(float)

# Vérifier les dimensions
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

from sklearn.preprocessing import LabelEncoder

# Encoder les variables catégoriques
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Features (8 colonnes uniquement)
feature_cols = [
    'age', 'sexe_encoded', 'temperature', 'tension_sys',
    'toux', 'fatigue', 'maux_tete', 'region_encoded'
]

X = df[feature_cols]
y = df['diagnostic']

print(f"Features : {X.shape}")
print(f"Cible : {y.shape}")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Entraînement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Modèle entraîné !")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred):.2%}")

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred, labels=model.classes_))

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("Modèle sauvegardé.")

# Rechargement
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

patients = [
    {'age': 20, 'sexe': 'F', 'temperature': 36.5, 'tension_sys': 120,
     'toux': False, 'fatigue': False, 'maux_tete': False, 'region': 'Dakar'},

    {'age': 35, 'sexe': 'M', 'temperature': 40.0, 'tension_sys': 110,
     'toux': True, 'fatigue': True, 'maux_tete': True, 'region': 'Dakar'},

    {'age': 65, 'sexe': 'M', 'temperature': 38.0, 'tension_sys': 130,
     'toux': True, 'fatigue': True, 'maux_tete': False, 'region': 'Dakar'}
]

for p in patients:
    sexe_enc = le_sexe_loaded.transform([p['sexe']])[0]
    region_enc = le_region_loaded.transform([p['region']])[0]

    features = [
        p['age'],
        sexe_enc,
        p['temperature'],
        p['tension_sys'],
        int(p['toux']),
        int(p['fatigue']),
        int(p['maux_tete']),
        region_enc
    ]

    pred = model_loaded.predict([features])[0]
    print(f"Patient {p['age']} ans -> Diagnostic : {pred}")

print(f"Modèle rechargé : {type(model_loaded).__name__}")

# Nouveau patient
nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'region': 'Dakar'
}

sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    region_enc
]

diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]
proba_max = probas.max()

print("\n--- Résultat du pré-diagnostic ---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilité : {proba_max:.1%}")

print("\nProbabilités par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"  {classe:12s} : {proba:.1%} {bar}")

# Importance des features
importances = model_loaded.feature_importances_
for name, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
    print(f" {name:20s} : {imp:.3f}")