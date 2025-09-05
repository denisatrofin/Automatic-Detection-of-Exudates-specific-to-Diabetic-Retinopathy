import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
from skimage.filters import frangi
import tempfile

import numpy as np
from skimage import io
from skimage.color import rgb2gray
import skfuzzy as fuzz
from ipywidgets import interact, widgets
from IPython.display import display

from sklearn.cluster import KMeans

from skimage.morphology import remove_small_objects


from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score


from sklearn.metrics import confusion_matrix
import glob
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.ndimage import distance_transform_edt
import scipy.ndimage as ndimage     # şi apoi ndimage.distance_transform_edt(...)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from skimage.measure import label, regionprops
import numpy as np

st.set_page_config(page_title="Detectare Exsudate", layout="centered")
st.title("Detectare automată a exsudatelor în imagini ale fundului de ochi")

def load_image(path_image):
    if not os.path.exists(path_image):
        raise FileNotFoundError("Calea catre imagine nu este corecta!")
        
    image = cv2.imread(path_image)
    
    if image is None:
        raise ValueError("Eroare in incarcarea imaginii!")
    return image


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


def padded_fov_mask(img_rgb, margin_ratio=0.01, blur_k=15):
    
    H, W = img_rgb.shape[:2]

    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    _, bin0 = cv2.threshold(blurred, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(bin0)
    if n_lbl > 1:
        idx_max = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (lbl == idx_max).astype(np.uint8)
    else:
        mask = bin0 // 255

    margin = int(margin_ratio * min(H, W))
    if margin:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (2*margin+1, 2*margin+1))
        mask = cv2.erode(mask, ker, 1)

    return mask          # 0 / 1

# Preprocesare
# =======================================================
def rgb_to_yiq(loaded_image_converted):
    # Convertim la float, normalizat în [0..1] pentru calcule
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


def yiq_to_rgb(Y, I, Q):
    
    R = Y + 0.956*I + 0.621*Q
    G = Y - 0.272*I - 0.647*Q
    B = Y - 1.105*I + 1.702*Q
    img_rgb = np.dstack([R, G, B])
    return img_rgb


def color_normalize_yiq(img_rgb, a=1.8, b=0.9, c=0.9, return_Ymod=False):

    Y, I, Q = rgb_to_yiq(img_rgb)
    Y_mod = a*Y - b*I - c*Q
    img_yiq_mod = yiq_to_rgb(Y_mod, I, Q)
    img_yiq_mod = np.clip(img_yiq_mod, 0.0, 1.0)
    img_uint8 = (img_yiq_mod * 255).astype(np.uint8)

    if return_Ymod:
        Y_mod_uint8 = np.clip(Y_mod, 0.0, 1.0)
        Y_mod_uint8 = (Y_mod_uint8 * 255).astype(np.uint8)
        return img_uint8, Y_mod_uint8

    return img_uint8


def preprocess_image(image_filtered):
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(9, 9))
    return clahe.apply(image_filtered)

def morphological_op(img, mask_fov):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))  # disc cu diametru 3 pixeli
    I = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # opening 
    
    # I_bg
    I_bg = cv2.blur(I, (51, 51))  # filtru medie 51x51

    # se calculeaza u = media intensitatilor
    u = np.mean(I)

    # se aplica formula I_ie = I - I_bg + u
    I_ie = I.astype(np.float32) - I_bg.astype(np.float32) + u
    # se normalizeaza doar în FOV
    I_ie = np.clip(I_ie, 0, 255).astype(np.uint8)
    I_ie[mask_fov == 0] = 0

    # fig = plt.figure(figsize=(8, 4))
    # ax1 = fig.add_subplot(1, 3, 1)
    # ax1.imshow(I, cmap='gray')                 
    # ax1.set_title("I")
    # ax1.axis("off")

    # ax2 = fig.add_subplot(1, 3, 2)
    # ax2.imshow(I_bg, cmap='gray')
    # ax2.set_title("I_bg")
    # ax2.axis("off")

    # ax3 = fig.add_subplot(1, 3, 3)
    # ax3.imshow(I_ie, cmap='gray')
    # ax3.set_title("I_ie")
    # ax3.axis("off")

    # plt.tight_layout()
    # plt.show()         
    # plt.close(fig)      

    return  I_ie, I, I_bg


# Detectie vase de sange si segmentare
# =======================================================
def remove_small_elements(image: np.ndarray, min_size: int) -> np.ndarray:
    # Detectăm componentele conectate din imaginea binară.
    # 'stats' conține pentru fiecare componentă: [left, top, width, height, area]
    components, output, stats, _ = cv2.connectedComponentsWithStats(
        image, connectivity=8)

    # se extrag vectorii cu dimensiunile (aria) componentelor 
    sizes = stats[1:, -1]      # aria fiecărei componente
    width = stats[1:, -3]      # lățimea fiecărei componente
    height = stats[1:, -2]     # înălțimea fiecărei componente

    # se ajusteaza indexul componentelor pentru a exclude fundalul
    components -= 1

    # Inițializăm o imagine goală pentru rezultat 
    result = np.zeros(output.shape, dtype=np.uint8)

    # Parcurgem fiecare componentă conectată (exceptând fundalul)
    for i in range(0, components):
        # Păstrăm doar componentele care respectă:
        # - o dimensiune minimă (aria >= min_size)
        # - și o lățime sau înălțime semnificativă (> 90 pixeli)
        if sizes[i] >= min_size and (width[i] > 90 or height[i] > 90):
            # Setăm pixelii corespunzători acestei componente la 255 în imaginea rezultat
            result[output == i + 1] = 255

    return result



# Detectie disc optic si segmentare
# =======================================================
# functie pentru incarcarea mastilor OD
def load_od_mask(image_name, od_masks_dir, target_size = (512, 512)):

    image_id = image_name.split('_')[-1].split('.')[0]  # extrage "01"
    od_filename = f'OD_{image_id}.jpg'
    od_mask_path = os.path.join(od_masks_dir, od_filename)
    od_mask = cv2.imread(od_mask_path, cv2.IMREAD_GRAYSCALE)
    if od_mask is None:
        raise FileNotFoundError(f"Masca OD nu a fost găsită pentru {image_name}")
    if od_mask.shape[:2] != (target_size[1], target_size[0]):
        od_mask = resize_with_padding(od_mask, target_size=target_size)
    return (od_mask > 0).astype(np.uint8)



def overlay_mask_outline(img_rgb, mask, color=(0, 0, 255), thickness=2):
    mask_u8   = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    outlined  = img_rgb.copy()
    cv2.drawContours(outlined, contours, -1, color, thickness,
                     lineType=cv2.LINE_AA)
    return outlined


# Segementare FCM 
# =======================================================
def fcm_subimage_segmentation(img_gray, img_prep, mask_fov, mode = "Screening", sub_size=(30, 40), c=2, global_k=0.1):
    # Dimensiuni imagine original

    H, W = img_gray.shape
    h_sub, w_sub = sub_size
    
    # matrice praguri locale 
    D_final = np.zeros((H, W), dtype=np.float64)

    # parcurgere fiecare subimagine 
    for y in range(0, H, h_sub):
        for x in range(0, W, w_sub):
            # coordonatele ferestrei
            y1 = y
            y2 = min(y + h_sub, H)
            x1 = x
            x2 = min(x + w_sub, W)

            # se extrage subimaginea curenta 
            sub_img = img_prep[y1:y2, x1:x2]

            # pregatire date pt fcm, vectorizare + normalizare 
            data = sub_img.flatten().astype(np.float64) / 255.0
            data = np.expand_dims(data, axis=0)  #

            # se aplica fcm pe subimagine 
            cntr_subImg, _, _, _, _, _, _ = fuzz.cluster.cmeans(
                data, c=c, m=2.0, error=1e-5, maxiter=1000
            )

            # calcul prag local
            if c == 2:
                threshold = np.mean(cntr_subImg)
            else:
                threshold = np.min(cntr_subImg)

            # Aplicare prag local pe subimagine
            D_final[y1:y2, x1:x2] = threshold

    # matricea D 
    D_final = cv2.blur(D_final, (10, 10))
    D_final = np.clip(D_final, 0, 1)
    print(f"[DBG] D_final min/max = {D_final.min():.3f}/{D_final.max():.3f}")

    # Calcul prag global pe intreaga imagine
    data_full = img_prep[img_prep > 0].astype(np.float64) / 255.0
    data_full = data_full[None, :]          

    # se aplica fccm pe imaginea intreaga 
    cntr_global, _, _, _, _, _, _ = fuzz.cluster.cmeans(
        data_full, c=c, m=2.0, error=1e-5, maxiter=1000
    )
    # matricea S 
    S = np.mean(cntr_global)
    S = np.clip(S, 0, 1)

    print(f"[DBG] S = {S:.3f}")

    S_final = np.full_like(D_final, S)

    # # Combinare praguri locale si globale cu formula din articol
    # T = global_k * S_final + (1 - global_k) * D_final

    # # se estimează “luminozitatea” imaginii doar în FOV
    #vals_T = T[mask_fov.astype(bool)]

    # # # folosim percentila 95
    # clip_min = np.percentile(vals_T, 88)      

    # # #  dacă fotografia e foarte întunecată, min = 0.3
    # clip_min = max(clip_min, 0.25)            
    # T = np.clip(T, clip_min, 1.0)

    if mode == "Screening":
        global_k = 0.1  # Prag local dominant (sensibilitate crescută)
        clip_percentile = 80  # Acceptă mai multe regiuni ca potențiale exsudate
        clip_min_abs = 0.2     # Prag minim mai indulgent
    else:  
        global_k = 0.1    # Prag global dominant (specificitate crescută)
        clip_percentile = 85  # Selectează doar zonele certe
        clip_min_abs = 0.28    # Prag minim mai strict

    T = global_k * S_final + (1 - global_k) * D_final
    vals_T = T[mask_fov.astype(bool)]  
    clip_min = max(np.percentile(vals_T, clip_percentile), clip_min_abs)
    T = np.clip(T, clip_min, 1.0)

    print(f"[DBG] T   min/max = {T.min():.3f}/{T.max():.3f}")

    # Segmentare binara finala
    segmentation = (img_gray.astype(np.float64)/255.0 > T).astype(np.uint8)

    return segmentation, D_final, S_final, T


# Extragerea trasaturilor
# =======================================================
def extract_features(green_channel, img_final_rgb, mask_fov, segmentation, combined_mask, dist_to_v):
    H, W = segmentation.shape
    
    X = []
    coords = []
    
    # Mean filter pe grayscale
    blurred_green = cv2.blur(green_channel, (3,3))

    gray_img = cv2.cvtColor(img_final_rgb, cv2.COLOR_RGB2GRAY)
    
    # Convertim imaginea RGB în HSV
    hsv_image = cv2.cvtColor(img_final_rgb, cv2.COLOR_RGB2HSV)
    
    # Mean filter pe canalele HSV
    hsv_blurred = cv2.blur(hsv_image, (3,3))
    
    # Morfologie OPEN o singură dată
    kernel = np.ones((3, 3), np.uint8)
    opened_green = cv2.morphologyEx(green_channel, cv2.MORPH_OPEN, kernel)
    
    # Gradient o singură dată
    grad_x = cv2.Sobel(green_channel, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel( green_channel, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    cand_mask  = (segmentation == 1)  & (dist_to_v   > 3) & (combined_mask.astype(bool)) & (mask_fov.astype(bool))
    ys, xs = np.where(cand_mask)
    
    for y, x in zip(ys, xs):
        if y-1 < 0 or y+2 > H or x-1 < 0 or x+2 > W:
            continue
        
        f1 = float(blurred_green[y, x])
        f2 = float(gray_img[y, x])
        f3 = float(hsv_blurred[y, x, 0])  # hue
        f4 = float(hsv_blurred[y, x, 1])  # saturation
        f5 = float(hsv_blurred[y, x, 2])  # value
        f6 = float(np.sum(green_channel[y-1:y+2, x-1:x+2]**2))  # energy
        f7 = float(np.std(opened_green[y-1:y+2, x-1:x+2]))  # deviație standard locală
        f8 = float(np.mean(gradient_magnitude[y-1:y+2, x-1:x+2]))  # gradient local
        
        X.append([f1, f2, f3, f4, f5, f6, f7, f8])
        coords.append((y, x))
    
    X = np.array(X, dtype=np.float32)

    print("\n Primele 10 caracteristici extrase:")
    print(X[:10])
    print("\n Primele 10 coordonate asociate:")
    print(coords[:10])
    
    return X, coords


def get_labels_from_gt_mask(coords, ex_mask):
    y_true = []
    for (y, x) in coords:
        y_true.append(int(ex_mask[y, x] > 0))  # 1 dacă e pixel etichetat ca exsudat, altfel 0
    return np.array(y_true)


def load_ex_mask(image_name, ex_dir, size=(512, 512)):

    image_id = image_name.split('_')[-1].split('.')[0]  
    ex_filename = f'IDRiD_{image_id}_EX.tif'
    ex_mask_path = os.path.join(ex_dir, ex_filename)
    ex_mask = cv2.imread(ex_mask_path, cv2.IMREAD_GRAYSCALE)
    if ex_mask is None:
        raise FileNotFoundError(f"Masca nu a fost gasita pentru {image_name}")
    ex_mask = resize_with_padding(ex_mask, target_size=size)

    return (ex_mask > 0).astype(np.uint8)


# def evaluate_pixelwise(prediction_map, ex_mask, mask_fov):
#     mask = mask_fov.astype(bool)
    
#     y_pred_full = (prediction_map > 0).astype(np.uint8)
#     y_true_full = (ex_mask > 0).astype(np.uint8)

#     y_pred = y_pred_full[mask]
#     y_true = y_true_full[mask]
    
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

#     accuracy = (tp + tn) / (tp + tn + fp + fn)
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0

#     return {
#         "TP": tp, "FP": fp, "FN": fn, "TN": tn,
#         "Accuracy": accuracy,
#         "Precision": precision,
#         "Recall": recall
#     }

def evaluate_pixelwise(prediction_map, ex_mask, mask_fov):
    mask = mask_fov.astype(bool)

    y_pred_full = (prediction_map > 0).astype(np.uint8)
    y_true_full = (ex_mask > 0).astype(np.uint8)

    y_pred = y_pred_full[mask]
    y_true = y_true_full[mask]

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]          
    ).ravel()

    accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Accuracy":  accuracy,
        "Precision": precision,
        "Recall":    recall
    }


def remove_elongated(binary_mask, axis_ratio_max=3.5):
    lbl = label(binary_mask, connectivity=2)
    clean = np.zeros_like(binary_mask)
    for r in regionprops(lbl):
        if r.minor_axis_length == 0:
            continue
        axis_ratio = r.major_axis_length / r.minor_axis_length
        if axis_ratio <= axis_ratio_max:          # păstrează doar pete compacte
            clean[lbl == r.label] = 1
    return clean.astype(np.uint8)

def postprocess_function(prediction_map, mask_fov, od_mask, dilated_vessels):
    
    pred = prediction_map.copy()          # uint8 0/255

    # 0.  INTERSECTEAZĂ cu FOV erodat 5 px
    fov_er = cv2.erode(mask_fov,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    pred &= fov_er

    # taie discul optic şi vasele dilatate cu 1 px
    pred &= ((1 - od_mask).astype(np.uint8) * 255)
    pred &= ((1 - dilated_vessels).astype(np.uint8) * 255)

    # se elimina componente mici 
    pred_bin = remove_small_objects(pred > 0, min_size=15)
    pred_bin = pred_bin.astype(np.uint8) * 255

    #  CLOSING morfologic 
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pred_bin = cv2.morphologyEx(pred_bin, cv2.MORPH_CLOSE, kernel)

    # umple golurile din interior 
    filled = ndimage.binary_fill_holes(pred_bin > 0).astype(np.uint8)
    filtered = remove_elongated(filled, axis_ratio_max=3.5)
    prediction_final = (filtered * 255).astype(np.uint8)

    return prediction_final


@st.cache_resource          
def load_svm():
    data = joblib.load("svm_exudate2.pkl")
    return data["model"]

model = load_svm()     


def process_image_and_evaluate(image_path, mode, model):

    width = 512
    height = 512
    od_masks_dir = r'C:\Users\Denisa\Desktop\licenta\od_mask_test'
    groundtruth_dir = r'C:\Users\Denisa\Desktop\licenta\3. Hard Exudates'

    # ===== Încarcă modelul antrenat =====
    # data_loaded = joblib.load("svm_exudate2.pkl")
    # model       = data_loaded["model"]
    # thr         = data_loaded["threshold"]

    name = os.path.basename(image_path)
    img = load_image(image_path)
    img_rgb0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = resize_with_padding(img_rgb0, target_size=(width, height))
    img_final_rgb = color_normalize_yiq(img_rgb, a=1.8, b=0.9, c=0.9, return_Ymod=False)  
    
    green_channel = img_final_rgb[:, :, 1]

    mask_fov  = padded_fov_mask(img_rgb, margin_ratio=0.01)

    # pentru vizualizare
    # outlined  = overlay_mask_outline(green_channel, mask_fov, color=(0,0,255), thickness=3)
    # cv2.imshow("FOV outline", cv2.cvtColor(outlined, cv2.COLOR_RGB2BGR))

    green_channel1 = green_channel * mask_fov                

    clahe_img = preprocess_image(green_channel1) 


    morph, _, _ = morphological_op(clahe_img, mask_fov)

    denoised = cv2.fastNlMeansDenoising(morph, None, 20)
    frangi_img = frangi(denoised)
    vessel_img = (frangi_img * 255).astype(np.uint8)
    _, binarized = cv2.threshold(vessel_img, 0, 255, cv2.THRESH_BINARY)
    final_vessels0 = remove_small_elements(binarized, 550).astype(np.uint8)
    final_vessels = cv2.bitwise_and(final_vessels0, final_vessels0, mask=mask_fov)  # se pastreaza doar pixelii din vasele de sânge care se află în interiorul zonei FOV (fara margini, padding)


    od_mask_bin = load_od_mask(name, od_masks_dir, target_size = (512, 512)) # se incarca mastile pt OD

    # elimină vasele din zona discului optic
    vessels_bin   = (final_vessels > 0).astype(np.uint8)
    masked_vessels = vessels_bin * (1 - od_mask_bin)
    final_vessels = (masked_vessels * 255).astype(np.uint8)

    vessel_mask = np.where(final_vessels == 255, 0, 255).astype(np.uint8)  # inversarea mastii - unde sunt vase = negru
    combined_mask = cv2.bitwise_and(vessel_mask, (1 - od_mask_bin)*255)   # masca care contine atat vasele, cat si discul optic (vase, OD = negru)
    dist_to_v     = ndimage.distance_transform_edt(vessels_bin == 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_vessels = cv2.dilate(vessels_bin, kernel, iterations=1)
    
    area_px = od_mask_bin.sum()
    r_est = int(np.sqrt(area_px / np.pi))
    delta = int(0.7 * r_est)
    ker_od = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*delta + 1, 2*delta + 1))
    od_mask = cv2.dilate(od_mask_bin, ker_od)

    mask_no_vas = ((1 - dilated_vessels) * (1 - od_mask)).astype(np.uint8) * 255
    mask_fcm = (mask_no_vas & (mask_fov*255)).astype(np.uint8)
    img_fcm = cv2.bitwise_and(morph, morph, mask=mask_fcm)

    segmentation, D_final, S_final, T = fcm_subimage_segmentation(img_fcm, img_fcm, mask_fov, mode = mode, sub_size=(30,40), c=2, global_k=0.1)

    X, coords = extract_features(green_channel, img_final_rgb, mask_fov, segmentation, combined_mask, dist_to_v)

    ex_mask = load_ex_mask(name, groundtruth_dir, size=(width, height))  # se incarca masca pt exsduate

    scores      = model.predict_proba(X)[:, 1]        


    if mode == "Screening":
        thr1 = 0.16
    else:
        thr1 = 0.70

    y_pred_all  = (scores > thr1).astype(int)

    prediction_map = np.zeros_like(green_channel, dtype=np.uint8)
    for (y, x), label in zip(coords, y_pred_all):
        if label == 1:
            prediction_map[y, x] = 255

    prediction_map = (prediction_map * mask_fov).astype(np.uint8)

    prediction_final = postprocess_function(prediction_map, mask_fov, od_mask, dilated_vessels)

    # === Evaluare ===
    metrics = evaluate_pixelwise(prediction_final, ex_mask, mask_fov)

    gt_cnt   = np.count_nonzero(ex_mask)        # câți pixeli sunt ≠ 0 în ground-truth
    pred_cnt = np.count_nonzero(prediction_final)  # câți pixeli ≠ 0 în predicție

    if gt_cnt == 0 and pred_cnt < 400:   # marja de eroare
        skip_metrics = True    # ochi sanatos
    else:
        skip_metrics = False

    # se afișează valorile
    print(f"Accuratețe: {metrics['Accuracy']:.4f}")
    print(f"Precizie: {metrics['Precision']:.4f}")
    print(f"Sensibilitate (Recall): {metrics['Recall']:.4f}")

    cm = np.array([[metrics['TN'], metrics['FP']],   #  rând 0
        [metrics['FN'], metrics['TP']]])  #  rând 1

    print("\nMatrice de confuzie [TN FP; FN TP]")
    print(cm)

    H, W = ex_mask.shape
    overlay = np.zeros((H, W, 3), dtype=np.uint8)

    for y in range(H):
        for x in range(W):
            if ex_mask[y, x] > 0 and prediction_final[y, x] > 0:
                overlay[y, x] = [0, 255, 255]   # galben = TP
            elif ex_mask[y, x] > 0:
                overlay[y, x] = [0, 255, 0]     # verde = FN (ratat)
            elif prediction_final[y, x] > 0:
                overlay[y, x] = [255, 0, 0]     # roșu = FP (fals pozitiv)


    return img_rgb, prediction_final, metrics, overlay, skip_metrics


st.subheader("Modul de funcționare")
mode = st.selectbox("Alege modul aplicației", ["Screening", "Monitorizare"])

uploaded_file = st.file_uploader("Încarcă o imagine", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Extrage numele original al fișierului (ex: IDRiD_55.jpg)
    original_name = uploaded_file.name

    # Creează o cale temporară cu acel nume
    temp_dir = tempfile.gettempdir()
    image_path = os.path.join(temp_dir, original_name)

    # se salvează imaginea cu numele original
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.read())


    try:
        img_rgb, prediction_final, metrics, overlay, skip_metrics = process_image_and_evaluate(image_path, mode, model)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Imagine originală RGB")
            st.image(img_rgb, channels="RGB", use_container_width=True)

        with col2:
            st.subheader("Mască exsudate detectate")
            st.image(prediction_final, clamp=True, use_container_width=True)

        # with col3:
        #     st.subheader("Mască exsudate detectate")
        #     st.image(overlay, clamp=True, use_container_width=True)

        st.subheader("Rezultate evaluare")

        if skip_metrics:
            st.success("Nu au fost detectate exsudate! Ochiul este sanatos.")
        else:
            if mode == "Screening":
                st.write(f"**Accuracy**: {metrics['Accuracy']:.3f}")
                st.write(f"**Recall**: {metrics['Recall']:.3f}")

            elif mode == "Monitorizare":
                st.write(f"**Accuracy**: {metrics['Accuracy']:.3f}")
                st.write(f"**Precision**: {metrics['Precision']:.3f}")
            st.error("Au fost detectate exsudate! Ochiul sufera de o forma de retinopatie diabetica.")


    except Exception as e:
        st.error(f"Eroare la procesare: {str(e)}")


