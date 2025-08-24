# api/main.py — fast & tolerant (rule-based parser + pgvector JOIN)
import os, time, json, re, unicodedata
from typing import List, Optional
from functools import lru_cache

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from psycopg_pool import ConnectionPool
from sentence_transformers import SentenceTransformer

# ====== CONFIG ======
DB_URL = "postgresql://neondb_owner:npg_LoMRfn9gqI0A@ep-noisy-darkness-a1wr0h4z-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Dùng 1 model nhỏ cho tốc độ (muốn mượt hơn thì đổi CHAT_MODEL = "qwen2.5:7b-instruct-q4_0")
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
    limit: int = 10
    page: int = 1
    semantic_query: Optional[str] = None

# ====== STARTUP ======
DISTRICTS: list[str] = []
WARDS: list[str] = []

@app.on_event("startup")
def _setup():
    # tạo extension & load từ vựng quận/phường cho parser rule-based
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS unaccent;")
                cur.execute("SELECT DISTINCT district FROM boarding_zones WHERE district IS NOT NULL LIMIT 5000")
                globals()["DISTRICTS"] = [r[0] for r in cur.fetchall()]
                cur.execute("SELECT DISTINCT ward FROM boarding_zones WHERE ward IS NOT NULL LIMIT 10000")
                globals()["WARDS"] = [r[0] for r in cur.fetchall()]
        print(f"[VOCAB] districts={len(DISTRICTS)} wards={len(WARDS)}")
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
    """Trả literal chuỗi pgvector để bind: %s::vector"""
    vec = _embed_model.encode([text], normalize_embeddings=True)[0].tolist()
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

def widen_price(pmin: Optional[float], pmax: Optional[float]):
    if pmin is not None and pmax is not None and pmin == pmax:
        delta = max(500_000, int(pmin * 0.15))  # ±500k hoặc 15%
        return pmin - delta, pmax + delta
    return pmin, pmax

# ====== RULE-BASED PARSER (siêu nhanh, không gọi LLM) ======
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def _contains(term: str, text: str) -> bool:
    return _strip_accents(term.lower()) in _strip_accents(text.lower())

def _parse_price(msg: str):
    txt = _strip_accents(msg.lower())
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*(tr|trieu|million)', txt)  # 2.5tr / 3 triệu
    if m:
        val = float(m.group(1).replace(",", "."))
        v = val * 1_000_000
        return max(0, v*0.85), v*1.15
    m = re.search(r'(\d[\d .]{5,})\s*(?:d|vnd)?', txt)  # 2 500 000 / 2500000
    if m:
        v = float(re.sub(r"[ .]", "", m.group(1)))
        return max(0, v*0.85), v*1.15
    return None, None

def _find_one(cands: list[str], msg: str) -> Optional[str]:
    for c in cands:
        if c and _contains(c, msg):
            return c
    return None

def parse_rule_based(message: str) -> dict:
    district = _find_one(DISTRICTS, message)
    ward     = _find_one(WARDS, message)
    pmin, pmax = _parse_price(message)
    amens = []
    for k in ["nội thất", "noi that", "máy lạnh", "may lanh", "máy giặt", "may giat", "wifi"]:
        if _contains(k, message):
            amens.append("nội thất" if "noi that" in _strip_accents(k) else k)
    return {
        "province": None,
        "district": district,
        "ward": ward,
        "price_min": pmin,
        "price_max": pmax,
        "area_min": None,
        "area_max": None,
        "amenities": list(set(amens)),
        "targets": [],
        "env_types": [],
        "semantic_query": message
    }

# ====== CORE ======
def parse_core(message: str) -> dict:
    # Dùng rule-based trước (0.01–0.05s)
    rb = parse_rule_based(message)
    if rb.get("district") or rb.get("ward") or (rb.get("price_min") and rb.get("price_max")) or rb["amenities"]:
        return rb
    # Fallback LLM (ít khi cần)
    system = ("Bạn chỉ trích xuất JSON cho truy vấn phòng trọ Việt Nam. "
              "Chỉ trả JSON với khóa: province,district,ward,price_min,price_max,area_min,area_max,amenities,targets,env_types,semantic_query.")
    prompt = f'Người dùng: "{message}"\nNếu thấy "tầm/khoảng ~X triệu", suy ra ±15%.'
    try:
        txt = ollama_generate(PARSE_MODEL, system, prompt, as_json=True, timeout=40)
        return json.loads(txt)
    except Exception:
        return rb

def search_core(p: Search) -> List[dict]:
    # KHÔNG chặn nhầm: coi NULL như true
    where = ["COALESCE(bz.is_visible, false) = false"]
    args: List = []

    def like_unaccent(col: str) -> str:
        return f"unaccent({col}) ILIKE unaccent(%s)"

    if p.province: where.append(like_unaccent("bz.province")); args.append(f"%{p.province}%")
    if p.district: where.append(like_unaccent("bz.district")); args.append(f"%{p.district}%")
    if p.ward:     where.append(like_unaccent("bz.ward"));     args.append(f"%{p.ward}%")
    if p.price_min is not None: where.append("bz.expected_price >= %s"); args.append(p.price_min)
    if p.price_max is not None: where.append("bz.expected_price <= %s"); args.append(p.price_max)
    if p.area_min  is not None: where.append("bz.area >= %s"); args.append(p.area_min)
    if p.area_max  is not None: where.append("bz.area <= %s"); args.append(p.area_max)
    if p.amenities:
        where.append("""EXISTS (SELECT 1 FROM boarding_zone_amenities a
                                WHERE a.boarding_zone_id = bz.id AND a.amenity_name = ANY(%s))""")
        args.append(p.amenities)
    if p.targets:
        where.append("""EXISTS (SELECT 1 FROM boarding_zone_targets t
                                WHERE t.boarding_zone_id = bz.id AND t.target_group = ANY(%s))""")
        args.append(p.targets)
    if p.env_types:
        where.append("""EXISTS (SELECT 1 FROM boarding_zone_environment e
                                WHERE e.boarding_zone_id = bz.id AND e.environment_type = ANY(%s))""")
        args.append(p.env_types)

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

                # JOIN bảng gốc để dùng bz.embedding (tránh lỗi 'embedding' không tồn tại trong filtered)
                sql = f"""
                WITH filtered AS (
                  {base}
                ),
                vec AS (
                  SELECT f.id, 1 - (bz.embedding <=> %s::vector) AS semantic_score
                  FROM filtered f
                  JOIN boarding_zones bz ON bz.id = f.id
                  ORDER BY bz.embedding <=> %s::vector
                  LIMIT 150
                )
                SELECT f.id, f.name, f.expected_price, f.area, f.province, f.district, f.ward,
                       f.address, f.description, f.created_at, v.semantic_score
                FROM filtered f
                JOIN vec v USING (id)
                ORDER BY v.semantic_score DESC, f.created_at DESC
                LIMIT %s OFFSET %s
                """
                cur.execute(sql, args + [qvec, qvec, p.limit, (p.page - 1) * p.limit])
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
            semantic_query=parsed.get("semantic_query") or msg,
        )
        items = search_core(params)
        t2 = timing("chat.search", t1)

        # context ngắn để LLM trả lời nhanh
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
