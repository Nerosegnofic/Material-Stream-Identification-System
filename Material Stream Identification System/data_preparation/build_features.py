import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from skimage.feature import local_binary_pattern, hog
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

IMG_SIZE = (128, 128)

COLOR_BINS = (8, 8, 8)
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_TYPE = "uniform"
HOG_CELL_SIZE = (8, 8)
HOG_BLOCK_SIZE = (2, 2)
HOG_ORIENTS = 9

random.seed(42)
np.random.seed(42)


def cv2_to_pil(img):
    if img is None:
        return None
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(img):
    arr = np.asarray(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def load_images_from_folder(folder_path):
    cats = sorted([
        d for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d))
    ])

    imgs, labels = [], []

    for i, cls in enumerate(cats):
        path = os.path.join(folder_path, cls)
        for f in os.listdir(path):
            p = os.path.join(path, f)
            img = cv2.imread(p)
            if img is None:
                continue
            imgs.append(img)
            labels.append(i)

    return imgs, labels, cats


# ---------------- PIL AUGMENTATIONS ----------------

def rotate_image(image):
    angle = random.uniform(-45, 45)
    return image.rotate(angle, fillcolor=(128, 128, 128))


def horizontal_flip(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def vertical_flip(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)


def random_scale(image):
    s = random.uniform(0.7, 1.3)
    w, h = image.size
    nw, nh = int(w * s), int(h * s)

    scaled = image.resize((nw, nh), Image.LANCZOS)

    if s > 1:
        x = (nw - w) // 2
        y = (nh - h) // 2
        return scaled.crop((x, y, x + w, y + h))
    else:
        bg = Image.new(
            "RGB",
            (w, h),
            tuple(random.randint(50, 200) for _ in range(3))
        )
        bg.paste(scaled, ((w - nw) // 2, (h - nh) // 2))
        return bg


def color_jitter(image):
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.5, 1.5))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.5, 1.5))
    image = ImageEnhance.Color(image).enhance(random.uniform(0.5, 1.5))
    return image


def gaussian_blur(image):
    return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 3)))


def random_crop_resize(image):
    w, h = image.size
    r = random.uniform(0.6, 0.9)
    cw, ch = int(w * r), int(h * r)

    x = random.randint(0, w - cw)
    y = random.randint(0, h - ch)

    crop = image.crop((x, y, x + cw, y + ch))
    return crop.resize((w, h), Image.LANCZOS)


def add_noise(image):
    arr = np.asarray(image, np.float32)
    noise = np.random.normal(0, random.uniform(5, 25), arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def cutout(image):
    img = image.copy()
    d = ImageDraw.Draw(img)
    w, h = img.size

    for _ in range(random.randint(1, 3)):
        cw = random.randint(w // 10, w // 4)
        ch = random.randint(h // 10, h // 4)
        x = random.randint(0, w - cw)
        y = random.randint(0, h - ch)
        d.rectangle(
            [x, y, x + cw, y + ch],
            fill=random.choice([
                (0, 0, 0),
                (128, 128, 128),
                tuple(random.randint(0, 255) for _ in range(3))
            ])
        )
    return img

def extract_features(image):
    image = cv2.resize(image, IMG_SIZE)

    color_feat = color_histogram(image)
    hog_feat = extract_hog(image)
    lbp_feat = extract_lbp(image)

    return np.concatenate([color_feat, hog_feat, lbp_feat]).astype(np.float32)

def perspective_transform(image):
    w, h = image.size
    c = [
        1 + random.uniform(-0.1, 0.1),
        random.uniform(-0.15, 0.15),
        random.uniform(-w * 0.05, w * 0.05),
        random.uniform(-0.15, 0.15),
        1 + random.uniform(-0.1, 0.1),
        random.uniform(-h * 0.05, h * 0.05),
        random.uniform(-5e-4, 5e-4),
        random.uniform(-5e-4, 5e-4),
    ]
    return image.transform(
        (w, h),
        Image.PERSPECTIVE,
        c,
        Image.BICUBIC,
        fillcolor=(128, 128, 128)
    )


def simulate_lighting(image):
    t = random.choice(["bright", "dim", "shadow"])

    if t == "bright":
        image = ImageEnhance.Brightness(image).enhance(random.uniform(1.3, 1.8))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 0.9))
        return image

    if t == "dim":
        return ImageEnhance.Brightness(image).enhance(random.uniform(0.4, 0.7))

    arr = np.asarray(image, np.float32)
    w, h = image.size
    mask = np.ones((h, w), np.float32)

    if random.random() < 0.5:
        for i in range(w):
            mask[:, i] *= 0.5 + i / w * 0.5
    else:
        for i in range(h):
            mask[i, :] *= 0.5 + i / h * 0.5

    for c in range(3):
        arr[:, :, c] *= mask

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def color_shift(image):
    arr = np.asarray(image, np.float32)
    arr[:, :, 0] *= random.uniform(0.8, 1.2)
    arr[:, :, 1] *= random.uniform(0.8, 1.2)
    arr[:, :, 2] *= random.uniform(0.8, 1.2)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def motion_blur(image):
    k = random.randint(3, 7)
    if random.random() < 0.5:
        kernel = ImageFilter.Kernel((k, 1), [1] * k, scale=k)
    else:
        kernel = ImageFilter.Kernel((1, k), [1] * k, scale=k)
    return image.filter(kernel)



def augment_image(img, count):
    base = cv2_to_pil(img)
    if base is None:
        return []

    ops = [
        rotate_image,
        random_scale,
        color_jitter,
        gaussian_blur,
        random_crop_resize,
        add_noise,
        cutout,
        perspective_transform,
        simulate_lighting,
        color_shift,
        motion_blur
    ]

    out = []

    for _ in range(count):
        tmp = base.copy()

        if random.random() < 0.05:
            out.append(pil_to_cv2(tmp.resize(IMG_SIZE)))
            continue

        if random.random() < 0.3:
            tmp = horizontal_flip(tmp)
        if random.random() < 0.1:
            tmp = vertical_flip(tmp)

        for op in random.sample(ops, random.randint(1, 3)):
            if random.random() < 0.85:
                tmp = op(tmp)

        tmp = tmp.resize(IMG_SIZE, Image.LANCZOS)
        out.append(pil_to_cv2(tmp))

    return out


def balance_dataset(images, labels, target):
    buckets = {}
    for im, lb in zip(images, labels):
        buckets.setdefault(lb, []).append(im)

    new_imgs, new_lbls = [], []

    for lb, imgs in buckets.items():
        for im in imgs:
            new_imgs.append(im)
            new_lbls.append(lb)

        if len(imgs) >= target:
            continue

        need = target - len(imgs)
        for _ in range(need):
            base = random.choice(imgs)
            aug = augment_image(base, 1)
            if aug:
                new_imgs.append(aug[0])
                new_lbls.append(lb)

    return new_imgs, new_lbls



def extract_hog(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return hog(
        g,
        orientations=HOG_ORIENTS,
        pixels_per_cell=HOG_CELL_SIZE,
        cells_per_block=HOG_BLOCK_SIZE,
        block_norm="L2-Hys",
        feature_vector=True
    ).astype(np.float32)


def extract_lbp(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(g, LBP_POINTS, LBP_RADIUS, LBP_TYPE)
    bins = LBP_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def color_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        COLOR_BINS, [0, 180, 0, 256, 0, 256]
    )
    return cv2.normalize(hist, hist).flatten()


def create_feature_matrix(images, labels):
    feats = []
    for img in tqdm(images, desc="extracting"):
        feats.append(extract_features(img))
    return np.vstack(feats), np.asarray(labels, np.int32)


def save_npz(X, y, path):
    np.savez_compressed(path, X=X, y=y)


def load_npz(path):
    d = np.load(path)
    return d["X"], d["y"]


def build_feature_dataset(data_path, target, out_file, balance=True):
    imgs, labels, classes = load_images_from_folder(data_path)

    if balance:
        imgs, labels = balance_dataset(imgs, labels, target)

    X, y = create_feature_matrix(imgs, labels)
    save_npz(X, y, out_file)
    return X, y, classes


if __name__ == "__main__":
    DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    TARGET = 500
    OUT = "features.npz"

    X, y, names = build_feature_dataset(DATA, TARGET, OUT)
    X2, y2 = load_npz(OUT)