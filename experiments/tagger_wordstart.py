"""
Full-vocab tagger using the tokenizer's OWN word-boundary signal.

No wordnet / no dict / no hand regex on meaning. The only structural signal is
the model's native BPE marker: a token starting with 'Ġ' (space) is a
WORD-START token = the model treats it as a standalone word, not a mid-word
fragment. This is the tokenizer's own internal notion of "word", zero external
resource.

Gate: keep vocab tokens whose raw BPE piece starts with 'Ġ' AND whose decoded
surface is a single alphabetic word. (alphabetic check via str method .isalpha,
not a lookup table.) Then centered scoring + embedding NMS.
"""
import sys, os, glob, json
import numpy as np
import mlx.core as mx
from PIL import Image
from transformers import Qwen2VLImageProcessor

MODEL_DIR=os.environ.get("JINA_OMNI_NANO_DIR","/Volumes/vault/ai-models/jinaai/jina-embeddings-v5-omni-nano-mlx")
sys.path.insert(0, MODEL_DIR)
from utils import load_model
IMAGE_TOKEN_ID=128259
HERE=os.path.dirname(__file__)

GT={"cat.jpg":{"cat","couch","remote","television","sofa","kitten","plush"},
    "livingroom_tv.jpg":{"television","chair","table","couch","furniture","room","fireplace","lamp","dresser"},
    "cat_shoe.jpg":{"cat","shoe","sneaker","kitten","sock"},
    "zebra.jpg":{"bear","grass","zebra"},
    "bus.jpg":{"sign","pole","road","tree"},
    "tennis.jpg":{"snow","ski","mountain","skier","snowboard","hill","sled"},
    "skate.jpg":{"kitchen","refrigerator","oven","stove","cabinet","cupboard","dresser"}}

def img_vec(model, proc, tok, path):
    out=proc(images=Image.open(path).convert("RGB"), return_tensors="np")
    grid=np.array(out["image_grid_thw"]).astype(np.int32)
    t,h,w=[int(x) for x in grid[0]]; n=(t*h*w)//4
    e=model.encode_image(mx.array(out["pixel_values"].astype(np.float32)),mx.array(grid),
                         mx.array([[IMAGE_TOKEN_ID]*n]),mx.array([[1]*n]))[0].astype(mx.float32)
    return np.array((e/mx.linalg.norm(e)).tolist())

def main():
    m=load_model(MODEL_DIR); m.switch_task("retrieval")
    proc=Qwen2VLImageProcessor.from_pretrained(MODEL_DIR)
    d=np.load(os.path.join(HERE,"full_vocab_emb.npz"), allow_pickle=True)
    E=d["E"].astype(np.float32); strings=list(d["strings"])
    En=E/np.linalg.norm(E,axis=1,keepdims=True)

    # tokenizer raw pieces -> word-start gate
    tj=json.load(open(os.path.join(MODEL_DIR,"tokenizer.json")))
    vocab=tj["model"]["vocab"]           # piece -> id
    id2piece={v:k for k,v in vocab.items()}
    V=len(strings)
    gate=np.zeros(V, dtype=bool)
    for tid in range(V):
        piece=id2piece.get(tid,"")
        if not piece.startswith("Ġ"):    # must be a word-start token
            continue
        surf=piece[1:]
        if len(surf)>=3 and surf.isalpha() and surf.isascii() and surf.islower():
            gate[tid]=True
    print(f"word-start alphabetic tokens: {gate.sum()}/{V}")

    BG=np.load(os.path.join(HERE,"bg_full.npy"))
    mu=BG.mean(0,keepdims=True)

    base=os.path.join(HERE,"testimg"); names=list(GT)
    IMG=np.stack([img_vec(m.model,proc,m.tokenizer,os.path.join(base,n)) for n in names])
    Cen=IMG@E.T - mu
    Cen[:, ~gate]=-1e9

    def nms(order, keep=8, tau=0.6):
        kept=[]; kv=[]
        for j in order:
            v=En[j]
            if any(float(v@k)>=tau for k in kv): continue
            kept.append(j); kv.append(v)
            if len(kept)>=keep: break
        return kept

    hit=0; hit3=0
    for i,name in enumerate(names):
        order=np.argsort(-Cen[i])[:300]
        kept=nms(order)
        tags=[strings[j].strip().lower() for j in kept]
        h=len(GT[name]&set(tags[:5]))>0; hit+=h
        h3=len(GT[name]&set(tags[:3]))>0; hit3+=h3
        print(f"\n===== {name}  GT={sorted(GT[name])}  {'HIT' if h else 'miss'} =====")
        print("   ", ", ".join(tags))
    print(f"\n[hit@5]={hit}/{len(names)}  [hit@3]={hit3}/{len(names)}")

if __name__=="__main__":
    main()
