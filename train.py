"""To run train.py
python train.py --train_dir train_dir --model_out model.keras --labels_out labels.json
"""


import os, json, random
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import argparse


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", type=str, default="train_dir")
parser.add_argument("--model_out", type=str, default="model.h5")
parser.add_argument("--labels_out", type=str, default="labels.json")
parser.add_argument("--char_h", type=int, default=64)
parser.add_argument("--char_w", type=int, default=64)
parser.add_argument("--min_char_width", type=int, default=6)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--val_fraction", type=float, default=0.10)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# Seed
random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# Utilities
def list_images(folder):
    p = Path(folder)
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    return sorted([str(fp) for fp in p.rglob("*") if fp.suffix.lower() in exts])

def label_from_filename(fp):
    return Path(fp).name[:2]

def to_grayscale(img):
    if img is None:
        return None
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# Segmentation
def segment_lines(gray):
    h = gray.shape[0]
    _, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,3))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE,kernel,iterations=1)
    proj = np.sum(closed,axis=1)
    if proj.max()==0: return [(0,h)]
    thresh=max(1,int(0.03*proj.max()))
    lines=[]
    in_line=False
    start=0
    for y,v in enumerate(proj):
        if v>thresh and not in_line:
            in_line=True
            start=y
        elif v<=thresh and in_line:
            end=y
            in_line=False
            if end-start>=6:
                lines.append((max(0,start-2),min(h,end+2)))
    if in_line:
        lines.append((start,h))
    return lines


def segment_words_from_line(line_img):
    gray = to_grayscale(line_img)
    _,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,3))
    dilated = cv2.dilate(th,kernel,iterations=1)
    contours,_ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes=[]
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w<8 or h<8: continue
        bboxes.append((x,y,w,h))
    bboxes=sorted(bboxes,key=lambda b:b[0])
    return [line_img[y:y+h,x:x+w] for (x,y,w,h) in bboxes]


def segment_chars_from_word(word_img,min_char_width=4):
    gray = to_grayscale(word_img)
    _,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cols=np.sum(th,axis=0)
    thresh=max(1,int(0.05*cols.max()))
    separators = cols<=thresh
    chars=[]
    in_char=False
    start=0
    for i,val in enumerate(separators):
        if (not val) and (not in_char):
            in_char=True
            start=i
        elif val and in_char:
            end=i
            in_char=False
            if end-start>=min_char_width:
                chars.append(word_img[:,start:end])
    if in_char:
        end=len(separators)
        if end-start>=min_char_width:
            chars.append(word_img[:,start:end])
    return chars


def resize_and_normalize_char(ch_img,target_h,target_w):
    if len(ch_img.shape)==2:
        ch=cv2.cvtColor(ch_img,cv2.COLOR_GRAY2RGB)
    else:
        ch=ch_img.copy()
    h,w=ch.shape[:2]
    scale=min(max(1e-6,target_w/w), max(1e-6,target_h/h))
    nw,nh=max(1,int(w*scale)), max(1,int(h*scale))
    resized=cv2.resize(ch,(nw,nh),interpolation=cv2.INTER_AREA)
    pad_left=(target_w-nw)//2
    pad_top=(target_h-nh)//2
    padded=255*np.ones((target_h,target_w,3),dtype=np.uint8)
    padded[pad_top:pad_top+nh,pad_left:pad_left+nw,:]=resized
    return padded.astype(np.float32)/255.0


# Augmentation for segmented CHARACTERS (64x64)
# We move this AFTER segmentation to avoid breaking the projection profiles
augmentation_char = tf.keras.Sequential([
    layers.RandomRotation(0.05), # Reduced from 0.15
    layers.RandomZoom(0.05, 0.05), # Reduced from 0.1
    layers.RandomContrast(0.1), # Reduced from 0.2
], name="char_augmentation")

@tf.function
def _augment_char_tensor(x):
    return augmentation_char(x)

def augment_char_numpy(ch_float):
    # ch_float is (64,64,3) in [0,1]
    t = tf.convert_to_tensor(ch_float[None, ...], dtype=tf.float32)
    aug = _augment_char_tensor(t)
    return aug[0].numpy()


# Load dataset and split images + duplication 
train_files=list_images(args.train_dir)
if len(train_files)==0: raise SystemExit("No training images found")
labels=sorted({label_from_filename(f) for f in train_files})
label_to_index={lab:i for i,lab in enumerate(labels)}

X,y=[],[]

for fp in tqdm(train_files,desc="Processing train images"):
    img=cv2.imread(fp)
    if img is None: continue
    H,W=img.shape[:2]
    # Split into 3 vertical patches
    third = W // 3
    patches = [img[:, :third], img[:, third:2*third], img[:, 2*third:]]
    
    # Process only original patches for segmentation
    # We augment the resulting CHARACTERS, not the input strips
    for patch in patches:
        gray=to_grayscale(patch)
        lines=segment_lines(gray)
        for y1,y2 in lines:
            line_img=patch[y1:y2,:]
            words=segment_words_from_line(line_img)
            if not words: words=[line_img]
            for w in words:
                chars=segment_chars_from_word(w,args.min_char_width)
                if not chars:
                    # No sub-chars found, take whole word/blob
                    ch=resize_and_normalize_char(w,args.char_h,args.char_w)
                    target_idx = label_to_index[label_from_filename(fp)]
                    
                    # Add Original
                    X.append(ch)
                    y.append(target_idx)
                    
                    # Add Augmented Copy
                    X.append(augment_char_numpy(ch))
                    y.append(target_idx)
                else:
                    for c in chars:
                        ch=resize_and_normalize_char(c,args.char_h,args.char_w)
                        target_idx = label_to_index[label_from_filename(fp)]

                        # Add Original
                        X.append(ch)
                        y.append(target_idx)

                        # Add Augmented Copy
                        X.append(augment_char_numpy(ch))
                        y.append(target_idx)

X=np.array(X,dtype=np.float32)
y=np.array(y,dtype=np.int32)

# Shuffle
perm=np.random.permutation(len(X))
X=X[perm]
y=y[perm]

y_cat=to_categorical(y,num_classes=len(labels))

# Train/val split
val_count=max(1,int(args.val_fraction*len(X)))
X_val,y_val=X[:val_count],y_cat[:val_count]
X_train,y_train=X[val_count:],y_cat[val_count:]

# Class weights
counts=Counter(y.tolist())
class_weight={i:(len(y)/(len(counts)*counts[i])) for i in counts}


# Angular Diversity Regularizer 
# Maximizes the minimum angular distance to the nearest neighbor
class AngularDiversityRegularizer(regularizers.Regularizer):
    def __init__(self, factor=0.01):
        self.factor = float(factor)
    def __call__(self, w):
        # Normalize weights to get unit vectors
        w_norm = tf.nn.l2_normalize(w, axis=0)
        # Compute Cosine Similarity Matrix
        gram = tf.matmul(tf.transpose(w_norm), w_norm)
        
        # Mask diagonal to ensure self-similarity is not the max
        # Gram values are [-1, 1]. Subtracting 2 makes diagonal -1.
        # This effectively ignores the diagonal when looking for max similarity.
        diag_mask = 2.0 * tf.eye(tf.shape(gram)[0])
        gram_masked = gram - diag_mask
        
        # Find max similarity for each class (its nearest, most confusing neighbor)
        max_sim = tf.reduce_max(gram_masked, axis=1)
        
        # Convert to angle (acos), clamping for stability
        tau = 0.99999
        max_sim_clamped = tf.clip_by_value(max_sim, -tau, tau)
        min_angles = tf.acos(max_sim_clamped)
        
        # We want to MAXIMIZE the mean angle.
        # So we MINIMIZE negative mean angle.
        return self.factor * (-tf.reduce_mean(min_angles))
        
    def get_config(self):
        return {'factor': self.factor}


# Build CNN model (Optimized Standard Conv2D)
def build_model(input_shape,num_classes):
    inp=layers.Input(shape=input_shape)
    # Slight in-model augmentation for robustness
    x = layers.RandomRotation(0.02)(inp)
    x = layers.RandomTranslation(0.02,0.02)(x)
    
    # Standard Conv2D is more robust for features than SeparableConv2D on this data
    # We use fewer filters to keep it CPU-friendly but accurate
    for filters in [32, 64, 128, 128]: # Reduced depth/width slightly for speed but kept Conv2D
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPool2D()(x)
        x = layers.Dropout(0.2)(x) 
        
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dense(512,activation="relu")(x)
    x=layers.BatchNormalization()(x)
    x=layers.Dropout(0.4)(x)
    # Apply Angular Diversity Constraint here
    out=layers.Dense(num_classes, activation="softmax", 
                     kernel_regularizer=AngularDiversityRegularizer(0.01))(x) # Reduced factor
    return models.Model(inp,out)

model=build_model((args.char_h,args.char_w,3),len(labels))
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"]) 


# Callbacks
ckpt=callbacks.ModelCheckpoint(args.model_out,monitor="val_accuracy",save_best_only=True,verbose=1)
early=callbacks.EarlyStopping(monitor="val_accuracy",patience=12,restore_best_weights=True,verbose=1)
rlr=callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=4,verbose=1)


# Train
history=model.fit(
    X_train,y_train,
    validation_data=(X_val,y_val),
    epochs=args.epochs,
    batch_size=args.batch_size,
    shuffle=True,
    class_weight=class_weight,
    callbacks=[ckpt,rlr,early]
)


# Save model & labels-
model.save(args.model_out)
with open(args.labels_out,"w") as f:
    json.dump(labels,f,indent=2)
print(f"Saved model to {args.model_out} and labels to {args.labels_out}")

