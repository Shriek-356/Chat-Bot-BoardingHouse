# api/main.py — fast & tolerant (rule-based parser + FTS + pgvector hybrid rank)
import os, time, json, re, unicodedata
from typing import List, Optional
from functools import lru_cache

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from psycopg_pool import ConnectionPool
from sentence_transformers import SentenceTransformer

# ====== CONFIG ======
DB_URL = (
    "postgresql://neondb_owner:npg_LoMRfn9gqI0A@ep-noisy-darkness-a1wr0h4z-pooler.ap-southeast-1.aws.neon.tech/neondb"
    "?sslmode=require"
    "&connect_timeout=15"
    "&hostaddr=13.228.184.177"
    "&keepalives=1&keepalives_idle=30&keepalives_interval=10&keepalives_count=5"
)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# Ngưỡng semantic (0..1). Đặt 0.0 nếu không muốn chặn.
SEM_MIN_SIM = float(os.getenv("SEM_MIN_SIM", "0.15"))

# Dùng 1 model nhỏ cho tốc độ
PARSE_MODEL = CHAT_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b-instruct-q4_0")

GEN_PARSE_OPTS = {"temperature": 0.0,  "num_predict": 120, "keep_alive": "30m", "num_thread": os.cpu_count() or 4}
GEN_CHAT_OPTS  = {"temperature": 0.25, "num_predict": 96,  "keep_alive": "30m", "num_thread": os.cpu_count() or 4}

# Embedding model (384 chiều)
_embed_model = SentenceTransformer("intfloat/multilingual-e5-small")

# DB pool
pool = ConnectionPool(conninfo=DB_URL, min_size=1, max_size=10)

app = FastAPI(title="Boarding Chatbot API")

# ====== MODELS ======
class Search(BaseModel):
    province: Optional[str] = None
    district: Optional[str] = None
    ward: Optional[str] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    area_min: Optional[float] = None
    area_max: Optional[float] = None
    amenities: Optional[List[str]] = None
    targets: Optional[List[str]] = None
    env_types: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    limit: int = 10
    page: int = 1
    semantic_query: Optional[str] = None

# ====== STARTUP ======
PROVINCES: List[str] = []
DISTRICTS: List[str] = []
WARDS: List[str] = []

@app.on_event("startup")
def _setup():
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS unaccent;")
                cur.execute("SELECT DISTINCT province FROM boarding_zones WHERE province IS NOT NULL LIMIT 500")
                globals()["PROVINCES"] = [r[0] for r in cur.fetchall()]
                cur.execute("SELECT DISTINCT district FROM boarding_zones WHERE district IS NOT NULL LIMIT 5000")
                globals()["DISTRICTS"] = [r[0] for r in cur.fetchall()]
                cur.execute("SELECT DISTINCT ward FROM boarding_zones WHERE ward IS NOT NULL LIMIT 10000")
                globals()["WARDS"] = [r[0] for r in cur.fetchall()]
        print(f"[VOCAB] provinces={len(PROVINCES)} districts={len(DISTRICTS)} wards={len(WARDS)}")
    except Exception as e:
        print("[WARN] startup:", e)

# ====== UTILS ======
def timing(label: str, t0: float) -> float:
    t1 = time.perf_counter()
    print(f"[TIMING] {label}: {t1 - t0:.2f}s")
    return t1

def ollama_generate(model: str, system: str, prompt: str, *, as_json=False, timeout=60) -> str:
    payload = {
        "model": model,
        "system": system or "",
        "prompt": prompt,
        "options": GEN_PARSE_OPTS if as_json else GEN_CHAT_OPTS,
        "stream": False,
    }
    if as_json: payload["format"] = "json"
    t0 = time.perf_counter()
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    print(f"[OLLAMA] {model} {r.status_code} in {time.perf_counter()-t0:.2f}s")
    r.raise_for_status()
    return r.json().get("response", "")

@lru_cache(maxsize=512)
def embed_vec_literal(text: str) -> str:
    vec = _embed_model.encode([text], normalize_embeddings=True)[0].tolist()
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

def widen_price(pmin: Optional[float], pmax: Optional[float]):
    if pmin is not None and pmax is not None and pmin == pmax:
        delta = max(500_000, int(pmin * 0.15))  # ±500k hoặc 15%
        return pmin - delta, pmax + delta
    return pmin, pmax

# ====== PARSER RULES ======
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def _contains(term: str, text: str) -> bool:
    return _strip_accents(term.lower()) in _strip_accents(text.lower())

def _parse_price(msg: str):
    txt = _strip_accents(msg.lower())
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*(tr|trieu|million)', txt)
    if m:
        val = float(m.group(1).replace(",", "."))
        v = val * 1_000_000
        return max(0, v*0.85), v*1.15
    m = re.search(r'(\d[\d .]{5,})\s*(?:d|vnd)?', txt)
    if m:
        v = float(re.sub(r"[ .]", "", m.group(1)))
        return max(0, v*0.85), v*1.15
    return None, None

def _find_one(cands: List[str], msg: str) -> Optional[str]:
    for c in cands:
        if c and _contains(c, msg):
            return c
    return None

# --- GEO NORMALIZATION ---
GEO_PREFIX_RE = re.compile(
    r'\b('
    r'q\.?|quan|quận|h\.?|huyen|huyện|tp|t\.p\.?|thanh pho|thành phố|'
    r'thi xa|tx|phuong|p\.?|phường'
    r')\b',
    flags=re.IGNORECASE
)

def _canon_geo(s: str) -> str:
    s = _strip_accents((s or "").lower())
    s = GEO_PREFIX_RE.sub(" ", s)
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _find_one_geo(cands: List[str], msg: str) -> Optional[str]:
    ms = _canon_geo(msg)
    if not ms:
        return None
    for c in cands:
        cs = _canon_geo(c or "")
        if not cs:
            continue
        if cs == ms or cs in ms or ms in cs:
            return c
    return None

# ====== CORE ======
# Domain phrases → chỉ giữ keyword có nghĩa
PET_SYNS = [
    "thú cưng", "nuôi thú cưng", "cho nuôi thú cưng",
    "pet", "pet friendly", "pet-friendly",
    "nuôi chó", "nuôi mèo", "cho nuôi chó", "cho nuôi mèo"
]
AMENITY_PHRASES = [
    "nội thất", "đầy đủ nội thất", "full nội thất", "full nt", "cơ bản nội thất",
    "máy lạnh", "điều hòa", "máy giặt", "tủ lạnh", "bàn học", "giường", "tủ quần áo",
    "ban công", "cửa sổ", "cửa sổ lớn", "ánh sáng tự nhiên",
    "thang máy",
    "gác", "gác lửng",
    "bếp", "nấu ăn", "cho nấu ăn", "khu bếp chung",
    "wc riêng", "toilet riêng", "nhà vệ sinh riêng",
    "chỗ để xe", "bãi xe", "giữ xe",
    "giờ giấc tự do", "tự do giờ giấc",
    "ở ghép", "share phòng", "share room",
    "gần trường", "gần đại học", "gần bệnh viện", "gần chợ", "gần bến xe"
]
VN_STOPWORDS = {
    "có","nào","ở","o","cho","không","khong","tìm","tim","phòng","phong","trọ","tro",
    "cần","can","giúp","giup","giá","gia","khoảng","khoang",
    "quận","quan","huyện","huyen","phường","phuong","tp","t p","thanh pho","thanh","pho",
    "tôi","toi","mình","minh","muốn","muon","cần tìm","can tim"
}

def _remove_geo_from_text(msg: str) -> str:
    ms = _canon_geo(msg)
    for x in (PROVINCES + DISTRICTS + WARDS):
        cx = _canon_geo(x or "")
        if not cx:
            continue
        ms = re.sub(rf'\b{re.escape(cx)}\b', ' ', ms)
    ms = re.sub(r'\s+', ' ', ms).strip()
    return ms

def _pick_phrases(msg: str, phrases: List[str]) -> List[str]:
    s = _strip_accents(msg.lower())
    out = []
    for p in phrases:
        if _strip_accents(p) in s:
            out.append(p)
    seen, uniq = set(), []
    for k in out:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq

def _extract_keywords(message: str) -> List[str]:
    kws = _pick_phrases(message, PET_SYNS)
    msg_wo_geo = _remove_geo_from_text(message)
    kws += [k for k in _pick_phrases(msg_wo_geo, AMENITY_PHRASES) if k not in kws]
    return kws[:3]

def parse_core(message: str) -> dict:
    province = _find_one_geo(PROVINCES, message)
    district = _find_one_geo(DISTRICTS, message)
    ward = _find_one_geo(WARDS, message)
    pmin, pmax = _parse_price(message)
    kw = _extract_keywords(message)

    return {
        "province": province,
        "district": district,
        "ward": ward,
        "price_min": pmin,
        "price_max": pmax,
        "area_min": None,
        "area_max": None,
        "amenities": [],
        "targets": [],
        "env_types": [],
        "keywords": kw,
        "semantic_query": message
    }

# Helper: canonize a SQL column (unaccent + drop geo prefixes + collapse spaces)
def _sql_canon(col: str) -> str:
    return (
        "regexp_replace("
        "  regexp_replace("
        f"    f_unaccent({col}),"
        "    '(\\y(q\\.?|quan|quận|h\\.?|huyen|huyện|tp|t\\.p\\.?|thanh pho|thành phố|thi xa|tx|phuong|p\\.?|phường)\\y)',"
        "    ' ', 'gi'"
        "  ),"
        "  '\\s+', ' ', 'g'"
        ")"
    )

def search_core(p: Search) -> List[dict]:
    # Hiển thị khoan dung: chỉ loại đúng false; true/NULL đều qua
    where = ["COALESCE(bz.is_visible, false) = false"]
    args: List = []

    def like_u(expr: str) -> str:
        return f"f_unaccent({expr}) ILIKE f_unaccent(%s)"

    # ===== Province strict (with address fallback) =====
    if p.province:
        canon = _canon_geo(p.province)
        col_canon = _sql_canon("bz.province")
        addr_canon = _sql_canon("bz.address")
        strict = f"(({col_canon} <> '') AND {col_canon} ILIKE %s)"
        fallback = f"(((bz.province IS NULL OR bz.province = '')) AND {addr_canon} ILIKE %s)"
        where.append(f"({strict} OR {fallback})")
        args += [f"%{canon}%", f"%{canon}%"]

    # ===== District strict (with address fallback) =====
    if p.district:
        canon = _canon_geo(p.district)
        col_canon = _sql_canon("bz.district")
        addr_canon = _sql_canon("bz.address")
        strict = f"(({col_canon} <> '') AND {col_canon} ILIKE %s)"
        fallback = f"(((bz.district IS NULL OR bz.district = '')) AND {addr_canon} ILIKE %s)"
        where.append(f"({strict} OR {fallback})")
        args += [f"%{canon}%", f"%{canon}%"]

    # ===== Ward strict (with address fallback) =====
    if p.ward:
        canon = _canon_geo(p.ward)
        col_canon = _sql_canon("bz.ward")
        addr_canon = _sql_canon("bz.address")
        strict = f"(({col_canon} <> '') AND {col_canon} ILIKE %s)"
        fallback = f"(((bz.ward IS NULL OR bz.ward = '')) AND {addr_canon} ILIKE %s)"
        where.append(f"({strict} OR {fallback})")
        args += [f"%{canon}%", f"%{canon}%"]

    # Numeric filters
    if p.price_min is not None:
        where.append("bz.expected_price >= %s"); args.append(p.price_min)
    if p.price_max is not None:
        where.append("bz.expected_price <= %s"); args.append(p.price_max)
    if p.area_min is not None:
        where.append("bz.area >= %s"); args.append(p.area_min)
    if p.area_max is not None:
        where.append("bz.area <= %s"); args.append(p.area_max)

    # --- Name/Description keywords (AND across keywords) ---
    if p.keywords:
        for kw in p.keywords:
            where.append("(" + " OR ".join([
                """to_tsvector('simple', f_unaccent(
                       coalesce(bz.name,'')||' '||coalesce(bz.description,'')||' '||coalesce(bz.address,'')
                   )) @@ websearch_to_tsquery('simple', f_unaccent(%s))""",
                like_u("bz.name"),
                like_u("bz.description"),
                like_u("bz.address")
            ]) + ")")
            args += [kw, f"%{kw}%", f"%{kw}%", f"%{kw}%"]

    base = f"""
      SELECT bz.id, bz.name, bz.expected_price, bz.area, bz.province, bz.district, bz.ward,
             bz.address, bz.description, bz.created_at
      FROM boarding_zones bz
      WHERE {' AND '.join(where)}
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            if p.semantic_query:
                qvec = embed_vec_literal(p.semantic_query)
                try:
                    cur.execute("SELECT set_config('ivfflat.probes', '8', true)")
                except Exception as e:
                    print("[WARN] cannot set ivfflat.probes:", e)

                sql = f"""
                WITH filtered AS (
                  {base}
                ),
                scored AS (
                  SELECT f.*,
                         ts_rank(
                           to_tsvector('simple', f_unaccent(
                             coalesce(f.name,'')||' '||coalesce(f.description,'')||' '||
                             coalesce(f.address,'')||' '||coalesce(f.district,'')||' '||coalesce(f.ward,'')
                           )),
                           plainto_tsquery('simple', f_unaccent(%s))
                         ) AS fts_score,
                         1 - (bz.embedding <=> %s::vector) AS emb_score
                  FROM filtered f
                  JOIN boarding_zones bz ON bz.id = f.id
                )
                SELECT * FROM scored
                WHERE (COALESCE(fts_score,0) > 0 OR emb_score >= %s)
                ORDER BY (COALESCE(fts_score,0)*0.4 + COALESCE(emb_score,0)*0.6) DESC,
                         created_at DESC
                LIMIT %s OFFSET %s
                """
                cur.execute(sql, args + [p.semantic_query, qvec, SEM_MIN_SIM, p.limit, (p.page - 1) * p.limit])
            else:
                sql = f"{base} ORDER BY created_at DESC LIMIT %s OFFSET %s"
                cur.execute(sql, args + [p.limit, (p.page - 1) * p.limit])

            cols = [c.name for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

# ====== ENDPOINTS ======
@app.post("/parse")
def parse(body: dict):
    t0 = time.perf_counter()
    out = parse_core((body.get("message") or "").strip())
    timing("parse", t0)
    return out

@app.post("/search")
def search(p: Search):
    t0 = time.perf_counter()
    pmin, pmax = widen_price(p.price_min, p.price_max)
    p = p.copy(update={"price_min": pmin, "price_max": pmax})
    items = search_core(p)
    timing("search", t0)
    return {"items": items}

@app.post("/chat")
def chat(body: dict):
    t_all = time.perf_counter()
    msg = (body.get("message") or "").strip()
    try:
        t0 = time.perf_counter()
        parsed = parse_core(msg)
        t1 = timing("chat.parse", t0)

        pmin, pmax = widen_price(parsed.get("price_min"), parsed.get("price_max"))
        params = Search(
            province=parsed.get("province"),
            district=parsed.get("district"),
            ward=parsed.get("ward"),
            price_min=pmin, price_max=pmax,
            area_min=parsed.get("area_min"),
            area_max=parsed.get("area_max"),
            amenities=parsed.get("amenities") or [],
            targets=parsed.get("targets") or [],
            env_types=parsed.get("env_types") or [],
            limit=5, page=1,
            keywords=parsed.get("keywords") or [],
            semantic_query=parsed.get("semantic_query") or msg,
        )
        items = search_core(params)
        t2 = timing("chat.search", t1)

        lines = []
        for r in items:
            desc = (r.get("description") or "")[:120]
            lines.append(f"- {r['name']} | {r.get('expected_price')}đ | {r.get('district','')} {r.get('ward','')}\n  {desc}...")
        context = "\n".join(lines) or "(không có kết quả phù hợp)"

        system = "Bạn là chatbot tư vấn trọ tiếng Việt, trả lời gọn, không bịa."
        prompt = f"""Yêu cầu: {msg}
Kết quả:
{context}

Viết 3–5 câu tư vấn, nêu vì sao phù hợp (giá/khu vực/tiện nghi), và gợi ý 1–2 lọc thêm."""
        answer = ollama_generate(CHAT_MODEL, system, prompt, as_json=False, timeout=90)
        timing("chat.answer", t2)
        timing("chat.total", t_all)
        return {"answer": answer, "items": items}
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
