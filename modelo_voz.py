
"""
=============================================================
 SISTEMA DE RECONOCIMIENTO DE VOZ
=============================================================
Division: 70% entrenamiento | 20% validacion | 10% prueba
"""

import os
import sys
import numpy as np
import librosa
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

DATASET_DIR  = "dataset"
MODEL_OUT    = "modelo_voz.pkl"
SCALER_OUT   = "scaler_voz.pkl"
ENCODER_OUT  = "encoder_voz.pkl"
SAMPLE_RATE  = 16000
N_MFCC       = 13
DURATION     = 1.5
TEST_SIZE    = 0.10
VAL_SIZE     = 0.222

COMANDOS = {
    "adelante":  "RECTA_10S",
    "atras":     "ATRAS_10S",
    "izquierda": "GIRO_90_IZQ",
    "derecha":   "GIRO_90_DER",
    "detener":   "STOP",
    "curva_izq": "CURVA_IZQ",
    "curva_der": "CURVA_DER",
    "ruido":     "IGNORAR",
}

def extraer_mfcc(ruta_archivo):
    try:
        audio, sr = librosa.load(ruta_archivo, sr=SAMPLE_RATE, duration=DURATION, mono=True)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        n_samples = int(SAMPLE_RATE * DURATION)
        if len(audio) < n_samples:
            audio = np.pad(audio, (0, n_samples - len(audio)))
        mfcc       = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=512, hop_length=160)
        mfcc_mean  = np.mean(mfcc, axis=1)
        mfcc_std   = np.std(mfcc, axis=1)
        delta      = librosa.feature.delta(mfcc)
        delta_mean = np.mean(delta, axis=1)
        return np.concatenate([mfcc_mean, mfcc_std, delta_mean])
    except Exception as e:
        print(f"  Advertencia: {ruta_archivo}: {e}")
        return None

def augmentar(audio, sr):
    versiones = []
    versiones.append(np.roll(audio, int(sr * 0.1)))
    versiones.append(audio + np.random.normal(0, 0.005, len(audio)))
    try:
        stretched = librosa.effects.time_stretch(audio, rate=0.9)
        n = int(sr * DURATION)
        stretched = stretched[:n] if len(stretched) >= n else np.pad(stretched, (0, n - len(stretched)))
        versiones.append(stretched)
    except:
        versiones.append(audio)
    return versiones

def extraer_con_augmentation(ruta_archivo):
    features_list = []
    try:
        audio, sr = librosa.load(ruta_archivo, sr=SAMPLE_RATE, duration=DURATION, mono=True)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        n = int(SAMPLE_RATE * DURATION)
        if len(audio) < n:
            audio = np.pad(audio, (0, n - len(audio)))
        feat = extraer_mfcc(ruta_archivo)
        if feat is not None:
            features_list.append(feat)
        for audio_aug in augmentar(audio, SAMPLE_RATE):
            mfcc  = librosa.feature.mfcc(y=audio_aug, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=512, hop_length=160)
            mean  = np.mean(mfcc, axis=1)
            std   = np.std(mfcc, axis=1)
            delta = librosa.feature.delta(mfcc)
            dmean = np.mean(delta, axis=1)
            features_list.append(np.concatenate([mean, std, dmean]))
    except Exception as e:
        print(f"  Advertencia augmentation: {e}")
    return features_list

def cargar_dataset():
    X, y = [], []
    print("\n  Cargando dataset...")
    print("-" * 55)
    clases = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    if not clases:
        print(f"  Error: No hay carpetas en '{DATASET_DIR}/'")
        sys.exit(1)
    for clase in sorted(clases):
        carpeta  = os.path.join(DATASET_DIR, clase)
        archivos = [f for f in os.listdir(carpeta) if f.lower().endswith(('.ogg','.wav','.mp3','.m4a','.flac'))]
        if not archivos:
            print(f"  Advertencia — {clase}: sin archivos de audio")
            continue
        count_orig, count_aug = 0, 0
        for archivo in archivos:
            ruta = os.path.join(carpeta, archivo)
            if clase == "ruido":
                feat = extraer_mfcc(ruta)
                if feat is not None:
                    X.append(feat); y.append(clase); count_orig += 1
            else:
                feats = extraer_con_augmentation(ruta)
                for feat in feats:
                    X.append(feat); y.append(clase)
                count_orig += 1; count_aug += len(feats) - 1
        print(f"  {clase:15s} — {count_orig} originales + {count_aug} aumentados = {count_orig+count_aug} muestras")
    print("-" * 55)
    print(f"  TOTAL: {len(X)} muestras, {len(set(y))} clases\n")
    return np.array(X), np.array(y)

def entrenar():
    print("\n  SISTEMA DE RECONOCIMIENTO DE VOZ")
    print("  Proyecto Final IA 2026 — Universidad Rafael Landívar")
    print("=" * 55)
    if not os.path.exists(DATASET_DIR):
        print(f"\n  Error: No existe la carpeta '{DATASET_DIR}/'")
        sys.exit(1)
    X, y = cargar_dataset()
    if len(X) == 0:
        print("\n  Error: No se cargaron datos.")
        sys.exit(1)

    encoder  = LabelEncoder()
    y_enc    = encoder.fit_transform(y)
    clases   = encoder.classes_
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("  Division del dataset: 70% entrenamiento | 20% validacion | 10% prueba")
    print("-" * 55)

    # Paso 1: apartar 10% prueba (el modelo nunca lo ve)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y_enc, test_size=TEST_SIZE, random_state=42, stratify=y_enc)

    # Paso 2: del 90% restante, separar 20% validacion
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=42, stratify=y_temp)

    total = len(X_scaled)
    print(f"  Entrenamiento : {len(X_train)} muestras ({len(X_train)/total*100:.0f}%)")
    print(f"  Validacion    : {len(X_val)} muestras ({len(X_val)/total*100:.0f}%)")
    print(f"  Prueba        : {len(X_test)} muestras ({len(X_test)/total*100:.0f}%)")
    print()

    # Modelo Base: MLP
    print("  Entrenando Modelo Base — MLP (Red Neuronal Densa)...")
    print("  Arquitectura: 39 -> 256 -> 128 -> 64 -> N_clases")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam',
        alpha=0.001, learning_rate='adaptive', max_iter=500, random_state=42,
        early_stopping=False, validation_fraction=0.0, verbose=False)
    mlp.fit(X_train, y_train)
    acc_mlp_val  = accuracy_score(y_val,  mlp.predict(X_val))
    acc_mlp_test = accuracy_score(y_test, mlp.predict(X_test))
    print(f"  MLP — Validacion: {acc_mlp_val*100:.1f}%  |  Prueba: {acc_mlp_test*100:.1f}%")

    # Modelo Secundario: SVM
    print("\n  Entrenando Modelo Secundario — SVM (kernel RBF)...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    acc_svm_val  = accuracy_score(y_val,  svm.predict(X_val))
    acc_svm_test = accuracy_score(y_test, svm.predict(X_test))
    print(f"  SVM — Validacion: {acc_svm_val*100:.1f}%  |  Prueba: {acc_svm_test*100:.1f}%")

    # Seleccionar mejor modelo por validacion
    if acc_mlp_val >= acc_svm_val:
        modelo_final, nombre_modelo = mlp, "MLP (Red Neuronal Densa)"
        acc_val_final, acc_test_final = acc_mlp_val, acc_mlp_test
    else:
        modelo_final, nombre_modelo = svm, "SVM"
        acc_val_final, acc_test_final = acc_svm_val, acc_svm_test

    print(f"\n  Modelo seleccionado: {nombre_modelo}")
    print(f"  Accuracy validacion : {acc_val_final*100:.1f}%")
    print(f"  Accuracy prueba     : {acc_test_final*100:.1f}%")

    # Metricas finales sobre conjunto de prueba
    y_pred = modelo_final.predict(X_test)

    # Colores ANSI
    VERDE    = "\033[92m"
    ROJO     = "\033[91m"
    AMARILLO = "\033[93m"
    CYAN     = "\033[96m"
    GRIS     = "\033[90m"
    BOLD     = "\033[1m"
    RESET    = "\033[0m"
    os.system('')

    print(f"\n  {BOLD}REPORTE DE METRICAS (conjunto de prueba 10%):{RESET}")
    print(f"  {GRIS}{'-'*55}{RESET}")
    report = classification_report(y_test, y_pred, target_names=clases, output_dict=True)
    print(f"  {'Clase':<14} {'Precision':>10} {'Recall':>8} {'F1':>8} {'N':>6}")
    print(f"  {GRIS}{'-'*50}{RESET}")
    for cls in clases:
        r = report[cls]
        f1 = r['f1-score']
        color = VERDE if f1 >= 0.95 else (AMARILLO if f1 >= 0.80 else ROJO)
        print(f"  {color}{cls:<14}{RESET}"
              f"  {r['precision']*100:>8.1f}%"
              f"  {r['recall']*100:>6.1f}%"
              f"  {color}{f1*100:>6.1f}%{RESET}"
              f"  {int(r['support']):>5}")

    # Matriz de confusion bonita
    cm = confusion_matrix(y_test, y_pred)
    abrev = [c[:6] for c in clases]
    col_w = 7
    print(f"\n  {BOLD}MATRIZ DE CONFUSION:{RESET}")
    print(f"  {GRIS}{'-'*55}{RESET}")
    # Encabezado columnas
    header = "  " + " " * 8
    for a in abrev:
        header += f"{a:>{col_w}}"
    print(f"{GRIS}{header}{RESET}")
    # Filas
    for i, row in enumerate(cm):
        linea = f"  {CYAN}{abrev[i]:<8}{RESET}"
        for j, val in enumerate(row):
            if i == j:
                color = VERDE if val > 0 else GRIS
            else:
                color = ROJO if val > 0 else GRIS
            linea += f"{color}{val:>{col_w}}{RESET}"
        total = row.sum()
        ok    = cm[i, i]
        pct   = ok / total * 100 if total > 0 else 0
        pcolor = VERDE if pct >= 95 else (AMARILLO if pct >= 80 else ROJO)
        linea += f"  {pcolor}{pct:5.1f}%{RESET}"
        print(linea)
    print(f"  {GRIS}(verde=correcto, rojo=error){RESET}")

    # Validacion cruzada 6-fold
    print("\n  Validacion cruzada estratificada (6-fold)...")
    skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    cv_scores = cross_val_score(modelo_final, X_scaled, y_enc, cv=skf, scoring='accuracy')
    print(f"  Accuracy promedio    : {cv_scores.mean()*100:.1f}%")
    print(f"  Desviacion estandar  : +/- {cv_scores.std()*100:.1f}%")
    print(f"  Por fold             : {[f'{s*100:.1f}%' for s in cv_scores]}")

    # Guardar
    joblib.dump(modelo_final, MODEL_OUT)
    joblib.dump(scaler,       SCALER_OUT)
    joblib.dump(encoder,      ENCODER_OUT)
    print(f"\n  Modelo guardado  : {MODEL_OUT}")
    print(f"  Scaler guardado  : {SCALER_OUT}")
    print(f"  Encoder guardado : {ENCODER_OUT}")
    print("\n  Listo para conectar con el robot!")
    print("=" * 55)
    return modelo_final, scaler, encoder, clases

def predecir_archivo(ruta, modelo, scaler, encoder):
    feat = extraer_mfcc(ruta)
    if feat is None:
        return None, 0.0
    feat_scaled = scaler.transform([feat])
    pred        = modelo.predict(feat_scaled)[0]
    proba       = modelo.predict_proba(feat_scaled)[0].max() if hasattr(modelo, 'predict_proba') else 1.0
    clase       = encoder.inverse_transform([pred])[0]
    return clase, proba

if __name__ == "__main__":
    modelo, scaler, encoder, clases = entrenar()
    print("\n  PRUEBA RAPIDA — ingrese rutas de archivos para probar")
    print("  (Ctrl+C para salir)")
    print("-" * 55)
    while True:
        try:
            ruta = input("\n  Ruta del archivo: ").strip().strip('"')
            if not ruta or not os.path.exists(ruta):
                print("  Archivo no encontrado.")
                continue
            clase, confianza = predecir_archivo(ruta, modelo, scaler, encoder)
            cmd = COMANDOS.get(clase, "DESCONOCIDO")
            print(f"  Prediccion: {clase:15s}  Comando: {cmd:15s}  Confianza: {confianza*100:.1f}%")
        except KeyboardInterrupt:
            print("\n\n  Saliendo...")
            break
