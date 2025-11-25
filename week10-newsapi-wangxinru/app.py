import streamlit as st
import requests
import pandas as pd
import numpy as np
import random
import math
from PIL import Image, ImageDraw, ImageFilter
import plotly.express as px
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="News + Emotion CrystalMix • wang xinru",
    page_icon="❄️",
    layout="wide"
)

nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

EMOTION_RGB = {
    "joy": (255,200,60),
    "love": (255,95,150),
    "pride": (255,160,90),
    "hope": (210,220,255),
    "calm": (120,170,255),
    "trust": (110,180,200),
    "curiosity": (150,200,255),
    "awe": (175,150,255),
    "nostalgia": (255,180,140),
    "surprise": (255,240,140),
    "mixed": (160,120,200),
    "anger": (245,60,60),
    "fear": (160,90,255),
    "sadness": (60,150,200),
    "anxiety": (100,130,180),
    "disgust": (120,160,90),
    "boredom": (180,180,180),
    "neutral": (160,160,160),
}

EMOTION_COLORNAME = {
    "joy": "Gold Ray",
    "love": "Rose Bloom",
    "pride": "Ember Orange",
    "hope": "Morning Haze",
    "calm": "Sky Blue",
    "trust": "Ocean Teal",
    "curiosity": "Mist Blue",
    "awe": "Violet Glow",
    "nostalgia": "Soft Amber",
    "surprise": "Lemon Light",
    "mixed": "Twilight Purple",
    "anger": "Flame Red",
    "fear": "Night Violet",
    "sadness": "Deep Aqua",
    "anxiety": "Storm Blue",
    "disgust": "Moss Green",
    "boredom": "Fog Gray",
    "neutral": "Silver Neutral",
}

def classify_emotion_expanded(row):
    neg=row["neg"]; neu=row["neu"]; pos=row["pos"]; compound=row["compound"]
    if compound>0.7 and pos>0.6: return "joy"
    if compound>0.6 and pos>0.5: return "love"
    if compound>0.5 and pos>0.4: return "pride"
    if compound>0.4 and pos>0.4: return "hope"
    if compound>0.2 and neu>0.4: return random.choice(["calm","trust","curiosity","awe","nostalgia"])
    if pos>0.3 and neu>0.4: return "surprise"
    if pos>0.25 and neg>0.25: return "mixed"
    if compound<-0.7 and neg>0.5: return "anger"
    if compound<-0.6: return "fear"
    if compound<-0.5: return "sadness"
    if compound<-0.3 and neg>0.3: return "anxiety"
    if compound<-0.2: return "disgust"
    if abs(compound)<0.05:
        return "boredom" if neu>0.8 else "neutral"
    return "neutral"

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1116; }
[data-testid="stSidebar"] { background: #0b0d12; }
h1,h2,h3,h4,h5,p,span,div,label { color: #e5e7eb !important; }
.glass { background: rgba(255,255,255,0.06); border-radius:16px; padding:14px; margin:10px 0; }
a { color: #8ab4ff !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    query = st.text_input("Keyword", "AI")
    page_size = st.slider("News Count", 10, 50, 20)
    compound_min, compound_max = st.slider("Compound Range", -1.0, 1.0, (-1.0, 1.0), step=0.01)
    auto_top3 = st.checkbox("Auto Top-3 Emotions", value=False)
    seed = st.slider("Crystal Seed", 1, 999, 123)
    layers = st.slider("Layers", 1, 8, 4)
    wobble = st.slider("Wobble", 0.0, 1.0, 0.25)
    blur_px = st.slider("Blur", 0, 20, 6)
    bg_color = (15,17,22)

def fetch_news(api_key, query, page_size):
    r=requests.get(
        "https://newsapi.org/v2/everything",
        params={"q":query,"pageSize":page_size,"apiKey":api_key,
                "sortBy":"publishedAt","language":"en"}
    )
    if r.status_code!=200: return pd.DataFrame()
    data=r.json().get("articles", [])
    if not data: return pd.DataFrame()
    df=pd.DataFrame(data)
    df["timestamp"]=pd.to_datetime(df["publishedAt"],errors="coerce")
    df["source"]=df["source"].apply(lambda x: x.get("name") if isinstance(x,dict) else x)
    df["text"]=df["title"].fillna("")+" "+df["description"].fillna("")
    return df[["timestamp","source","title","description","text","url"]]

api_key = st.secrets["NEWS_API_KEY"]
df = fetch_news(api_key, query, page_size)

if df.empty:
    st.warning("No news found.")
    st.stop()

vader = df["text"].apply(sia.polarity_scores)
df["neg"]=vader.apply(lambda x:x["neg"])
df["neu"]=vader.apply(lambda x:x["neu"])
df["pos"]=vader.apply(lambda x:x["pos"])
df["compound"]=vader.apply(lambda x:x["compound"])

df = df[(df["compound"]>=compound_min)&(df["compound"]<=compound_max)]
if df.empty:
    st.warning("No news after compound filtering.")
    st.stop()

df["emotion"] = df.apply(classify_emotion_expanded, axis=1)

emotion_counts = df["emotion"].value_counts()
unique_emotions = emotion_counts.index.tolist()

formatted_labels = {e:f"{e} ({EMOTION_COLORNAME.get(e,'')})" for e in unique_emotions}

if auto_top3:
    selected_emotions = emotion_counts.head(3).index.tolist()
else:
    selected_emotions = st.sidebar.multiselect(
        "Selected Emotions",
        options=list(formatted_labels.keys()),
        default=list(formatted_labels.keys()),
        format_func=lambda x: formatted_labels[x]
    )

df = df[df["emotion"].isin(selected_emotions)]
if df.empty:
    st.warning("No news for selected emotions.")
    st.stop()
def crystal_shape(cx, cy, r, wobble=0.25, sides_min=5, sides_max=10):
    n = random.randint(sides_min, sides_max)
    angles = np.linspace(0, 2*math.pi, n, endpoint=False)
    np.random.shuffle(angles)
    pts=[]
    for a in angles:
        rr = r*(1+wobble*(random.random()-0.5))
        pts.append((cx+rr*math.cos(a), cy+rr*math.sin(a)))
    pts.append(pts[0])
    return pts

def draw_polygon_soft(base, points, color, alpha=150, blur_px=6, edge_width=1):
    layer = Image.new("RGBA", base.size, (0,0,0,0))
    draw = ImageDraw.Draw(layer)
    draw.polygon(points, fill=(*color, alpha))
    if edge_width>0:
        draw.line(points, fill=(255,255,255,200), width=edge_width)
    layer = layer.filter(ImageFilter.GaussianBlur(blur_px))
    base.alpha_composite(layer)
    return base

def srgb_to_linear(arr):
    arr=arr/255.0
    return np.where(arr<=0.04045, arr/12.92, ((arr+0.055)/1.055)**2.4)

def linear_to_srgb(arr):
    arr=np.where(arr<=0.0031308, arr*12.92, 1.055*(arr**(1/2.4))-0.055)
    return np.clip(arr*255, 0, 255)

def apply_exposure(arr, ev):
    return arr*(2**ev)

def apply_white_balance(arr, temp, tint):
    r=arr[:,:,0]; g=arr[:,:,1]; b=arr[:,:,2]
    t=temp/100.0
    r*=1+0.1*t
    b*=1-0.1*t
    g*=1+0.05*tint
    return np.clip(np.stack([r,g,b],axis=-1),0,1)

def highlight_rolloff(arr, strength=0.3):
    return arr/(arr+strength)

def tonemap_aces(arr):
    a=2.51; b=0.03; c=2.43; d=0.59; e=0.14
    return np.clip((arr*(a*arr+b))/(arr*(c*arr+d)+e),0,1)

def adjust_contrast(arr, c):
    return np.clip((arr-0.5)*c+0.5,0,1)

def rgb_to_hsv(arr):
    mx=arr.max(axis=2); mn=arr.min(axis=2); df=mx-mn
    h=np.zeros_like(mx)
    mask=mx==arr[:,:,0]
    h[mask]=((arr[:,:,1]-arr[:,:,2])[mask]/df[mask])%6
    mask=mx==arr[:,:,1]
    h[mask]=((arr[:,:,2]-arr[:,:,0])[mask]/df[mask])+2
    mask=mx==arr[:,:,2]
    h[mask]=((arr[:,:,0]-arr[:,:,1])[mask]/df[mask])+4
    h[df==0]=0; h=h/6
    s=np.where(mx==0,0,df/mx)
    v=mx
    return h,s,v

def hsv_to_rgb(h,s,v):
    h=h*6; i=np.floor(h).astype(int)
    f=h-i; p=v*(1-s); q=v*(1-f*s); t=v*(1-(1-f)*s)
    i_mod=i%6
    out=np.zeros((h.shape[0],h.shape[1],3))
    idx=i_mod==0; out[idx]=np.stack([v,t,p],axis=-1)[idx]
    idx=i_mod==1; out[idx]=np.stack([q,v,p],axis=-1)[idx]
    idx=i_mod==2; out[idx]=np.stack([p,v,t],axis=-1)[idx]
    idx=i_mod==3; out[idx]=np.stack([p,q,v],axis=-1)[idx]
    idx=i_mod==4; out[idx]=np.stack([t,p,v],axis=-1)[idx]
    idx=i_mod==5; out[idx]=np.stack([v,p,q],axis=-1)[idx]
    return out

def adjust_saturation(arr, sat):
    h,s,v=rgb_to_hsv(arr)
    s=np.clip(s*sat,0,1)
    return hsv_to_rgb(h,s,v)

def gamma_correct(arr, g):
    return np.clip(arr**(1/g),0,1)

def split_toning(arr, sh, hi, balance=0.5, strength=0.25):
    lum=arr.mean(axis=2)
    sh_mask=lum<balance
    hi_mask=~sh_mask
    out=arr.copy()
    out[sh_mask]=np.clip(out[sh_mask]+strength*np.array(sh),0,1)
    out[hi_mask]=np.clip(out[hi_mask]+strength*np.array(hi),0,1)
    return out

def auto_brightness(arr, black=5, white=95, target=0.5):
    lum=arr.mean(axis=2).flatten()*255
    lo=np.percentile(lum,black)/255
    hi=np.percentile(lum,white)/255
    arr=(arr-lo)/(hi-lo)
    arr=np.clip(arr,0,1)
    g=target/arr.mean()
    arr=np.clip(arr*g,0,1)
    return arr

def apply_bloom(arr, radius=25, strength=0.25):
    img=(arr*255).astype(np.uint8)
    pil=Image.fromarray(img)
    blur=np.array(pil.filter(ImageFilter.GaussianBlur(radius)))/255.0
    return np.clip(arr*(1-strength)+blur*strength,0,1)

def apply_vignette(arr, strength=0.4):
    h,w,_=arr.shape
    y,x=np.ogrid[:h,:w]
    cy, cx = h/2, w/2
    dist=np.sqrt((x-cx)**2 + (y-cy)**2)
    maxd=np.sqrt(cx**2+cy**2)
    mask=1-strength*(dist/maxd)
    mask=np.clip(mask,0,1)
    return arr*mask[:,:,None]

def ensure_colorfulness(arr, min_sat=0.25):
    h,s,v=rgb_to_hsv(arr)
    s=np.where(s<min_sat, min_sat, s)
    return hsv_to_rgb(h,s,v)

def render_crystalmix(df, width=1920, height=1080, seed=123,
                      shapes_per_emotion=5, min_size=40, max_size=140,
                      wobble=0.25, blur_px=6, layers=4, bg_color=(15,17,22)):
    random.seed(seed); np.random.seed(seed)
    base=Image.new("RGBA",(width,height),bg_color+(255,))
    emotions=df["emotion"].value_counts().index.tolist()
    if not emotions: emotions=["joy","love","curiosity"]
    margin=120
    for _ in range(layers):
        for emo in emotions:
            color=EMOTION_RGB.get(emo,(150,150,150))
            for _ in range(shapes_per_emotion):
                cx=random.randint(margin,width-margin)
                cy=random.randint(margin,height-margin)
                r=random.randint(min_size,max_size)
                pts=crystal_shape(cx,cy,r,wobble)
                alpha=150+random.randint(-30,30)
                blur=blur_px+random.randint(-2,2)
                edge=1
                base=draw_polygon_soft(base, pts, color, alpha, blur, edge)
    arr=np.array(base)[:,:,:3].astype(np.float32)
    arr=srgb_to_linear(arr)
    arr=apply_exposure(arr,0.0)
    arr=apply_white_balance(arr,0.0,0.0)
    arr=highlight_rolloff(arr,0.3)
    arr=tonemap_aces(arr)
    arr=adjust_contrast(arr,1.1)
    arr=adjust_saturation(arr,1.15)
    arr=gamma_correct(arr,1.0)
    arr=split_toning(arr, sh=(0.0,0.05,0.25), hi=(0.25,0.15,0.0), balance=0.45, strength=0.25)
    arr=auto_brightness(arr, black=5, white=95, target=0.55)
    arr=apply_bloom(arr, radius=25, strength=0.22)
    arr=apply_vignette(arr, strength=0.35)
    arr=ensure_colorfulness(arr, min_sat=0.22)
    arr=linear_to_srgb(arr)
    return Image.fromarray(arr.astype(np.uint8))
st.header("News Articles")

for _, row in df.iterrows():
    st.markdown(f"""
    <div class='glass'>
        <h4>{row['title']}</h4>
        <p>{row['description']}</p>
        <p style='opacity:.7'>{row['source']} — {row['timestamp']}</p>
        <b>Emotion:</b> {row['emotion']} ({EMOTION_COLORNAME.get(row['emotion'],'')})<br>
        <a href="{row['url']}" target="_blank">Read more →</a>
    </div>
    """, unsafe_allow_html=True)

st.header("Source Distribution")
src=df["source"].value_counts().reset_index()
src.columns=["Source","Count"]
fig=px.bar(src,x="Source",y="Count",title="Articles by Source")
fig.update_layout(paper_bgcolor="#0f1116",plot_bgcolor="#0f1116",font_color="#e5e7eb")
st.plotly_chart(fig,use_container_width=True)

st.header("Emotion CrystalMix • 1920 × 1080")
crystal = render_crystalmix(
    df,
    width=1920,
    height=1080,
    seed=seed,
    shapes_per_emotion=5,
    min_size=40,
    max_size=140,
    wobble=wobble,
    blur_px=blur_px,
    layers=layers,
    bg_color=bg_color
)
st.image(crystal,use_container_width=True)
