import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import frangi

import numpy as np
from skimage import io
from skimage.color import rgb2gray
import skfuzzy as fuzz
from IPython.display import display

from skimage.morphology import remove_small_objects

from sklearn import svm
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import joblib
from scipy.ndimage import zoom

from sklearn.model_selection import cross_val_score
from scipy.ndimage import distance_transform_edt
import scipy.ndimage as ndimage     # şi apoi ndimage.distance_transform_edt(...)
from sklearn.model_selection import GroupKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from skimage.feature import graycomatrix, graycoprops 
from imblearn.over_sampling import SMOTE  
from imblearn.pipeline      import Pipeline     # pipeline compatibil cu samplere

from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay



def load_image(path_image):
    if not os.path.exists(path_image):
        raise FileNotFoundError("Calea catre imagine nu este corecta!")
        
    image = cv2.imread(path_image)
    
    if image is None:
        raise ValueError("Eroare in incarcarea imaginii!")
    return image

# Preprocesare
# =======================================================

def resize_with_padding(image, target_size=(512, 512)):
    h, w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=0)
    return padded_image


# functie de conversie rgb -> YIQ
def rgb_to_yiq(loaded_image_converted):
    # convertim la float, normalizat în [0..1] pentru calcule
    img_float = loaded_image_converted.astype(np.float32) / 255.0
    
    # Separăm canalele
    R = img_float[:,:,0]
    G = img_float[:,:,1]
    B = img_float[:,:,2]
    
    # Formula  RGB -> YIQ
    Y = 0.299*R + 0.587*G + 0.114*B
    I = 0.596*R - 0.274*G - 0.322*B
    Q = 0.212*R - 0.523*G + 0.311*B   
    return Y, I, Q

# functie de conversie YIQ -> RGB
def yiq_to_rgb(Y, I, Q):

    R = Y + 0.956*I + 0.621*Q
    G = Y - 0.272*I - 0.647*Q
    B = Y - 1.105*I + 1.702*Q

    img_rgb = np.dstack([R, G, B])
    return img_rgb


def color_normalize_yiq(img_rgb, a=1.8, b=0.9, c=0.9):

    # 1) RGB -> YIQ
    Y, I, Q = rgb_to_yiq(img_rgb)
    
    # 2) Ajustarea componentei Y
    Y_mod = a*Y - b*I - c*Q
    
    # 3) YIQ -> RGB cu Y_mod în loc de Y
    img_yiq_mod = yiq_to_rgb(Y_mod, I, Q)
    
    # 4) Clip și conversie la uint8
    img_yiq_mod = np.clip(img_yiq_mod, 0.0, 1.0)
    img_yiq_mod = (img_yiq_mod * 255).astype(np.uint8)
    
    return img_yiq_mod


def preprocess_image(image_filtered):
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(9, 9))
    return clahe.apply(image_filtered)

def morphological_op(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))  # disc cu diametru 3 pixeli
    I = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # opening 
    
    # I_bg
    I_bg = cv2.blur(I, (51, 51))  # filtru medie 51x51

    # se calculeaza u = media intensitatilor
    u = np.mean(I)

    # se aplica formula I_ie = I - I_bg + u
    I_ie = I.astype(np.float32) - I_bg.astype(np.float32) + u
    I_ie = np.clip(I_ie, 0, 255).astype(np.uint8)

    mask = I > 10
    I_ie[~mask] = 0

    return  I_ie, I, I_bg


# Detectie vase de sange si segmentare
# =======================================================
def remove_small_elements(image: np.ndarray, min_size: int) -> np.ndarray:

    # se detectează componentele conexe
    components, output, stats, _ = cv2.connectedComponentsWithStats(
        image, connectivity=8)

    # se extrag dimensiunile pentru fiecare componentă, ignorând fundalul
    sizes = stats[1:, -1]
    width = stats[1:, -3]
    height = stats[1:, -2]

    # ajustează numărul total de componente, ignorând fundalul
    components -= 1
    # se crează o imagine goală
    result = np.zeros(output.shape, dtype=np.uint8)

    for i in range(0, components):
        # păstrează doar componentele care au cel puțin min_size și fie lățimea, fie înălțimea mai mare decât 90 pixeli
        if sizes[i] >= min_size and (width[i] > 90 or height[i] > 90):
            # etichetează pixelii corespunzători componentei selectate în imaginea rezultată
            result[output == i + 1] = 255

    return result

# functie pentru incarcarea imaginilor cu OD
def load_od_mask(image_name, od_masks_dir, size=(512, 512)):
    # se extrage ID-ul din numele imaginii
    image_id = image_name.split('_')[-1].split('.')[0]  # extrage "01"
    # se construiește numele fișierului măștii OD corespunzătoare
    od_filename = f'OD_{image_id}.jpg'
    # se csreează calea completă către fișierul măștii OD
    od_mask_path = os.path.join(od_masks_dir, od_filename)
    od_mask = cv2.imread(od_mask_path, cv2.IMREAD_GRAYSCALE)
    if od_mask is None:
        raise FileNotFoundError(f"Masca OD nu a fost găsită pentru {image_name}")
    od_mask = cv2.resize(od_mask, size, interpolation=cv2.INTER_NEAREST)
    return (od_mask > 0).astype(np.uint8)

def padded_fov_mask(img_rgb, margin_ratio=0.01, blur_k=15):

    # se obțin dimensiunile imaginii 
    H, W = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # se aplică un blur Gaussian pentru netezirea zgomotului
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    # se binarizează imaginea cu Otsu pentru a obține o mască aproximativă a FOV
    _, bin0 = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # se găsește componentele conexe (zonele albe)
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(bin0)

    if n_lbl > 1:
        # selectează componenta conexă cu suprafața maximă (excluzând fundalul)
        idx_max = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (lbl == idx_max).astype(np.uint8)
    else:
        # dacă există doar fundal, folosește masca așa cum e
        mask = bin0 // 255

    # se calculează marginea în pixeli ca procent din dimensiunea imaginii
    margin = int(margin_ratio * min(H, W))

    if margin:
        # se creează un element structurant de formă eliptică, dimensiune în funcție de margine
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        # se erodează masca pentru a retrage marginile, eliminând zonele marginale instabile
        mask = cv2.erode(mask, ker, 1)

    return mask

# funcție utilizată pentru încărcarea măștilor de ground truth
def load_ex_mask(image_name, ex_dir, size=(512, 512)):

    image_id = image_name.split('_')[-1].split('.')[0]  
    ex_filename = f'IDRiD_{image_id}_EX.tif'
    ex_mask_path = os.path.join(ex_dir, ex_filename)
    ex_mask = cv2.imread(ex_mask_path, cv2.IMREAD_GRAYSCALE)
    if ex_mask is None:
        raise FileNotFoundError(f"Masca nu a fost gasita pentru {image_name}")
    ex_mask = resize_with_padding(ex_mask, target_size=size)

    return (ex_mask > 0).astype(np.uint8)

def overlay_mask_outline(img_rgb, mask, color=(0, 0, 255), thickness=2):
    mask_u8   = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    outlined  = img_rgb.copy()
    cv2.drawContours(outlined, contours, -1, color, thickness,
                     lineType=cv2.LINE_AA)
    return outlined


def extract_features(green_channel, img_final_rgb,
                     mask_fov, ex_mask, combined_mask, frangi_img,
                     dist_to_v, vessels_bin,
                     band_px=3, hard_ratio=0.30):

    H, W = green_channel.shape

    X = []
    coords = []
    labels = []

    # === Preprocesare ===
    blurred_green = cv2.blur(green_channel, (3, 3))
    gray_img = cv2.cvtColor(img_final_rgb, cv2.COLOR_RGB2GRAY)

    # Convertim imaginea RGB în HSV
    hsv_image = cv2.cvtColor(img_final_rgb, cv2.COLOR_RGB2HSV)
    
    # Mean filter pe canalele HSV
    hsv_blurred = cv2.blur(hsv_image, (3,3))

    opened_green = cv2.morphologyEx(green_channel, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    grad_x = cv2.Sobel(green_channel, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(green_channel, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # === 1. Coordonate pozitive din mască ===
    pos_coords = list(zip(*np.where(ex_mask == 1)))

    # ---------------------------------------------------
    # B. pixeli negativi

    # B1. hard negatives: bandă +/- 'band_px' în jurul vaselor
    band_mask   = (dist_to_v <= band_px) & (ex_mask == 0) & (mask_fov == 1)
    hard_coords = list(zip(*np.where(band_mask)))

    # B2. easy negatives: restul fundalului (fără vas & OD & exudat)
    easy_mask   = (combined_mask != 0) & (ex_mask == 0) & (dist_to_v > band_px) & (mask_fov ==1)
    easy_coords = list(zip(*np.where(easy_mask)))

    # ---------------------------------------------------
    # C. eșantionare la raport 30 % hard, 70 % easy
    n_pos       = len(pos_coords)
    n_hard_neg  = int(hard_ratio * n_pos)
    n_easy_neg  = n_pos - n_hard_neg          # păstrezi raport 1:1 total

    np.random.shuffle(hard_coords)
    np.random.shuffle(easy_coords)

    neg_coords  = hard_coords[:n_hard_neg] + easy_coords[:n_easy_neg]

    all_coords = pos_coords + neg_coords
    all_labels = [1] * len(pos_coords) + [0] * len(neg_coords)

    # === 3. Extragem trăsături pentru toate coordonatele ===
    for (y, x), label in zip(all_coords, all_labels):
        if y - 1 < 0 or y + 2 > H or x - 1 < 0 or x + 2 > W:
            continue

        f1 = float(blurred_green[y, x])
        f2 = float(gray_img[y, x])
        f3 = float(hsv_blurred[y, x, 0])  # hue
        f4 = float(hsv_blurred[y, x, 1])  # saturation
        f5 = float(hsv_blurred[y, x, 2])  # value
        f6 = float(np.sum(green_channel[y - 1:y + 2, x - 1:x + 2] ** 2))  # energy
        f7 = float(np.std(opened_green[y-1:y+2, x-1:x+2]))  # deviație standard locală
        f8 = float(np.mean(gradient_magnitude[y-1:y+2, x-1:x+2]))  # gradient local 

        X.append([f1, f2, f3, f4, f5, f6, f7, f8])  
        coords.append((y, x))
        labels.append(label)
        # de afisat all_labels

    X = np.array(X, dtype=np.float32)
    labels = np.array(labels, dtype=np.uint8)

    # === Debug ===
    print("\n Primele 10 caracteristici extrase:")
    print(X[:50])
    print(" Primele 10 coordonate:")
    print(coords[:10])
    print(" Distributie etichete:", np.unique(labels, return_counts=True))

    return X, coords, labels

def visualize_extracted_points(img_rgb, pos_coords, neg_coords, save_path=None):
    # Copiem imaginea originală pentru a nu o modifica direct
    img_copy = img_rgb.copy()

    # Marcam punctele de exsudate (pozitive) - galben
    for (y, x) in pos_coords:
        cv2.circle(img_copy, (x, y), 2, (255, 255, 0), -1)

    # Marcam punctele non-exsudate (negative) - albastru
    for (y, x) in neg_coords:
        cv2.circle(img_copy, (x, y), 2, (0, 0, 255), -1)

    # Afisare imagine
    plt.figure(figsize=(12, 12))
    plt.imshow(img_copy)
    plt.title("Puncte Exsudate (Galben) și Non-Exsudate (Albastru)")
    plt.axis("off")
    
    plt.show()


# funcție ce caută automat cei mai buni hiperparametri pentru un model SVM cu kernel RBF, folosind scalare și cross-validation pe grupuri
# X este matricea cu trăsăturile de intrare, y = vectoul de etichete reale
def gridsearch_svm(X, y, groups, cv_folds=5):
    print("\nPornim căutarea celor mai buni parametri SVM...")

    # Pipeline: scalare + SVM
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', probability=True, class_weight='balanced'))
    ])

    # Grid de parametri
    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__gamma': [0.001, 0.01, 0.1]
    }

    # Cross-validation pe grupuri (imagini)
    gkf = GroupKFold(n_splits=cv_folds)

    grid_search = GridSearchCV(
        pipe,
        param_grid,
        scoring='f1',  
        cv=gkf.split(X, y, groups),
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X, y)
    
    print(f"\n[INFO] Cea mai bună combinație: {grid_search.best_params_}")
    print(f"[INFO] Scor (F1) în validare: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

# funcție ce evaluează performanța modelului de clasificare folosind cross-validation pe grupuri și calculează metrici de performanță 
def evaluate(model, X, y, groups, n_splits=5, y_pred=None):
    # dacă nu există predicții, se calculează cu cross-validation pe grupuri
    if y_pred is None:
        gkf = GroupKFold(n_splits=n_splits)
        y_pred = cross_val_predict(
            model, X, y,
            cv=gkf.split(X, y, groups),
            n_jobs=-1
        )

    # calculează metricile de performanță
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    # afișează metricile și matricea de confuzie
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))

    return acc, prec, rec, f1

# funcție ce realizează graficul precizie-recall
def evaluate_pr_curve_from_scores(y_true, y_scores):
    # calculează punctele precision-recall și pragurile
    prec, rec, thr = precision_recall_curve(y_true, y_scores)
    # calculează Average Precision (AP)
    ap = average_precision_score(y_true, y_scores)

    # Plotează curba Precision-Recall
    plt.figure()
    plt.plot(rec, prec, label=f"OOF AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.show()

    return prec, rec, thr


def main():
    # setează directoarele cu date și măști
    data = r'C:\Users\Denisa\Desktop\licenta\train_data'
    ex_dir = r'C:\Users\Denisa\Desktop\licenta\ex_masks'
    od_masks_dir = r'C:\Users\Denisa\Desktop\licenta\od_mask'

    # listează toate imaginile din directorul de date
    testing_data = [x for x in os.listdir(data)]
    print("Numar imagini:", len(testing_data))

    # inițializează listele globale pentru trăsături, etichete și grupuri
    all_X_global = []
    all_y_global = []
    all_groups   = []

    width = 512
    height = 512

    for i, name in enumerate(testing_data):
        try:
            # construiește calea completă a imaginii
            image_path = os.path.join(data, name)
            print(name)

            # încarcă imaginea și convertește din BGR în RGB
            img = load_image(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # redimensionează imaginea cu padding la dimensiunea dorită
            img_rgb = resize_with_padding(img_rgb, target_size=(width, height))

        except Exception as e:
            # afișează eroarea dacă apare o problemă la încărcare
            print("Eroare la imaginea:", name, "\n", e)

        # Preprocesare

        # normalizează culoarea imaginii în spațiul YIQ
        img_final_rgb = color_normalize_yiq(img_rgb, a=1.8, b=0.9, c=0.9)
        # extrage canalul verde din imaginea normalizată
        green_channel = img_final_rgb[:, :, 1]

        # creează masca FOV cu o margine mică
        mask_fov = padded_fov_mask(img_rgb, margin_ratio=0.01)

        # opțional: se afișează conturul FOV 
        # outlined = overlay_mask_outline(img_rgb, mask_fov, color=(0,0,255), thickness=3)
        # cv2.imshow("FOV outline", cv2.cvtColor(outlined, cv2.COLOR_RGB2BGR))

        # aplică masca FOV peste canalul verde
        green_channel = green_channel * mask_fov

        # încarcă masca reală a exsudatelor 
        ex_masks = load_ex_mask(name, ex_dir, size=(width, height))

        # aplică CLAHE pe canalul verde
        clahe_img = preprocess_image(green_channel)
        # aplică operații morfologice 
        morph, _, _ = morphological_op(clahe_img)
        # aplică filtrare pentru reducerea zgomotului
        denoised = cv2.fastNlMeansDenoising(morph, None, 15)


        # Segmentare OD și vase de sânge

        # aplică filtrul Frangi pentru evidențierea vaselor de sânge
        frangi_img = frangi(denoised)
        # convertește rezultatul Frangi la imagine pe 8 biți
        vessel_img = (frangi_img * 255).astype(np.uint8)
        # binarizează imaginea vaselor
        _, binarized = cv2.threshold(vessel_img, 0, 255, cv2.THRESH_BINARY)

        # elimină elementele mici din masca vaselor
        final_vessels0 = remove_small_elements(binarized, 550).astype(np.uint8)
        # aplică masca FOV pentru a elimina chenarul exterior
        final_vessels = cv2.bitwise_and(final_vessels0, final_vessels0, mask=mask_fov)

        # încarcă masca discului optic (OD)
        od_mask_bin = load_od_mask(name, od_masks_dir, size=(width, height))
        # convertește la mască binară
        vessels_bin = (final_vessels > 0).astype(np.uint8)
        # elimină discul optic din masca vaselor
        masked_vessels = vessels_bin * (1 - od_mask_bin)
        # reconstruiește masca finală a vaselor
        final_vessels = (masked_vessels * 255).astype(np.uint8)

        # inversează masca vaselor (vasele devin 0)
        vessel_mask = np.where(final_vessels == 255, 0, 255).astype(np.uint8)
        # combină masca vaselor cu masca OD (zone fără vase și fără OD)
        combined_mask = cv2.bitwise_and(vessel_mask, (1 - od_mask_bin) * 255)

        # calculează distanța la cel mai apropiat vas (pentru hard negatives)
        dist_to_v = ndimage.distance_transform_edt(vessels_bin == 0)


        # Extragerea de trăsături pt clasificare
        X, coords, labels = extract_features(green_channel, img_final_rgb, mask_fov, ex_masks,combined_mask, frangi_img, dist_to_v, vessels_bin)


        # separare coordonate pozitive și negative
        pos_coords = [coord for coord, label in zip(coords, labels) if label == 1]
        neg_coords = [coord for coord, label in zip(coords, labels) if label == 0]

        # vizualizare puncte selectate pt antrenare (dacă se doreste)
        visualize_extracted_points(img_final_rgb, pos_coords, neg_coords)

        all_X_global.append(X)
        all_y_global.append(labels)
        all_groups.extend([name] * len(labels))
        print("Nr coordonate candidate:", len(coords))

    # combină toate trăsăturile extrase din toate imaginile
    X_final = np.vstack(all_X_global)   # (N, 8)
    # combină toate etichetele într-un singur vector
    Y_final = np.hstack(all_y_global)   # (N,)
    # combină grupurile într-un singur vector
    groups = np.array(all_groups)       # (N,)

    # verifică tipurile de date
    print(X_final.dtype)
    print(Y_final.dtype)

    # afișează numărul total de puncte
    print(f"\n Date combinate: {X_final.shape[0]} puncte în total.")

    # afișează distribuția claselor (0 vs. 1)
    print("Distribuție finală a claselor în Y_final:", np.unique(Y_final, return_counts=True))

    # se găsesc cei mai buni hiperparametri și se antrenează modelul SVM
    model = gridsearch_svm(X_final, Y_final, groups)

    # definește schema de cross-validation pe grupuri (ex. imagini)
    gkf = GroupKFold(n_splits=5)

    # generează probabilități OOF (out-of-fold) pentru fiecare exemplu
    proba_oof = cross_val_predict(
        model,               # modelul antrenat cu parametrii găsiți
        X_final,             # trăsăturile de intrare
        Y_final,             # etichetele reale (ground truth)
        cv=gkf.split(X_final, Y_final, groups),  # split pe grupuri
        method='predict_proba',   # obține probabilitatea clasei pozitive
        n_jobs=-1           # folosește toate nucleele CPU disponibile
    )[:, 1]  # păstrează doar coloana cu probabilitatea clasei 1

    # calculează curba precision-recall și găsește pragul optim
    prec, rec, thr = precision_recall_curve(Y_final, proba_oof)
    best_thr = thr[np.argmax(2 * prec * rec / (prec + rec + 1e-9))]

    # generează predicția finală OOF folosind pragul ales
    y_pred_oof = (proba_oof > best_thr).astype(int)

    # afișează metricile finale pe datele OOF
    evaluate(model, X_final, Y_final, groups, y_pred=y_pred_oof)

    # se trasează curba precision-recall
    evaluate_pr_curve_from_scores(Y_final, proba_oof)

    # salvează modelul final și pragul optim într-un fișier .pkl
    joblib.dump({'model': model, 'threshold': best_thr}, 'svm_exudate3.pkl')


if __name__ == '__main__':
    main() ;