"""
To run run.py
python run.py --test_dir test_dir --model model.keras --labels labels.json --out result.csv
"""

import os, json, csv
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from tensorflow.keras.models import load_model

parser=argparse.ArgumentParser()
parser.add_argument("--test_dir",required=True)
parser.add_argument("--model",required=True)
parser.add_argument("--labels",required=True)
parser.add_argument("--out",default="result.csv")
parser.add_argument("--char_h",type=int,default=64)
parser.add_argument("--char_w",type=int,default=64)
parser.add_argument("--min_char_width",type=int,default=6)
args=parser.parse_args()

# Load
# Load
from tensorflow.keras.layers import RandomRotation
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras import regularizers
import tensorflow as tf

# Define the custom regularizer used in training
class AngularDiversityRegularizer(regularizers.Regularizer):
    def __init__(self, factor=0.01):
        self.factor = float(factor)
    def __call__(self, w):
        w_norm = tf.nn.l2_normalize(w, axis=0)
        gram = tf.matmul(tf.transpose(w_norm), w_norm)
        
        diag_mask = 2.0 * tf.eye(tf.shape(gram)[0])
        gram_masked = gram - diag_mask
        
        max_sim = tf.reduce_max(gram_masked, axis=1)
        
        tau = 0.99999
        max_sim_clamped = tf.clip_by_value(max_sim, -tau, tau)
        min_angles = tf.acos(max_sim_clamped)
        
        return self.factor * (-tf.reduce_mean(min_angles))
    def get_config(self):
        return {'factor': self.factor}

# Monkey-patch RandomRotation to accept value_range
_original_init = RandomRotation.__init__
def _new_init(self, factor, value_range=None, **kwargs):
    if value_range is not None:
        print(f"Ignored value_range: {value_range}")
    _original_init(self, factor, **kwargs)
RandomRotation.__init__ = _new_init

with custom_object_scope({'AngularDiversityRegularizer': AngularDiversityRegularizer}):
    model=load_model(args.model)


with open(args.labels,"r") as f:
    labels=json.load(f)
index_to_label={i:lab for i,lab in enumerate(labels)}


def to_grayscale(img):
    if len(img.shape)==3:
        return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

# Reusing segmentation helpers same as train
def segment_lines(gray):
    h=gray.shape[0]
    _,th=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(1,3))
    closed=cv2.morphologyEx(th,cv2.MORPH_CLOSE,kernel,iterations=1)
    proj=np.sum(closed,axis=1)
    if proj.max()==0:return[(0,h)]
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
    if in_line: lines.append((start,h))
    return lines


def segment_words_from_line(line_img):
    gray=to_grayscale(line_img)
    _,th=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(15,3))
    dilated=cv2.dilate(th,kernel,iterations=1)
    contours,_=cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    bboxes=[]
    for cnt in contours:
        x,y,w,h=cv2.boundingRect(cnt)
        if w<8 or h<8: continue
        bboxes.append((x,y,w,h))
    bboxes=sorted(bboxes,key=lambda b:b[0])
    return [line_img[y:y+h,x:x+w] for (x,y,w,h) in bboxes]


def segment_chars_from_word(word_img,min_char_width=4):
    gray=to_grayscale(word_img)
    _,th=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cols=np.sum(th,axis=0)
    thresh=max(1,int(0.05*cols.max()))
    separators=cols<=thresh
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
    else: ch=ch_img.copy()
    h,w=ch.shape[:2]
    scale=min(max(1e-6,target_w/w), max(1e-6,target_h/h))
    nw,nh=max(1,int(w*scale)), max(1,int(h*scale))
    resized=cv2.resize(ch,(nw,nh),interpolation=cv2.INTER_AREA)
    pad_left=(target_w-nw)//2
    pad_top=(target_h-nh)//2
    padded=255*np.ones((target_h,target_w,3),dtype=np.uint8)
    padded[pad_top:pad_top+nh,pad_left:pad_left+nw,:]=resized
    return padded.astype(np.float32)/255.0


# Test
results=[]
correct=0
total=0

test_files=[f for f in os.listdir(args.test_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]

for f in tqdm(test_files,desc="Testing"):
    true_class=f[:2]
    img=cv2.imread(os.path.join(args.test_dir,f))
    if img is None: continue
    gray=to_grayscale(img)
    lines=segment_lines(gray)
    chars_imgs=[]
    for y1,y2 in lines:
        line_img=img[y1:y2,:]
        words=segment_words_from_line(line_img)
        if not words: words=[line_img]
        for w in words:
            chars=segment_chars_from_word(w,args.min_char_width)
            if not chars:
                ch=resize_and_normalize_char(w,args.char_h,args.char_w)
                chars_imgs.append(ch)
            else:
                for c in chars:
                    ch=resize_and_normalize_char(c,args.char_h,args.char_w)
                    chars_imgs.append(ch)
    if not chars_imgs:
        # no characters found: skip file but log it
        results.append([f,true_class,'NO_CHARS_FOUND'])
        continue
    X_chars=np.array(chars_imgs)
    preds=model.predict(X_chars,batch_size=32)
    pred_labels=np.argmax(preds,axis=1)
    pred_class=index_to_label[np.bincount(pred_labels).argmax()]
    results.append([f,true_class,pred_class])
    if pred_class==true_class: correct+=1
    total+=1

# Save CSV
with open(args.out,"w",newline="") as f:
    writer=csv.writer(f)
    writer.writerow(["filename","true_class","predicted_class"])
    writer.writerows(results)

accuracy=correct/total if total>0 else 0
print(f"Average Accuracy = {accuracy*100:.2f}%")
print(f"Results saved to {args.out}")
