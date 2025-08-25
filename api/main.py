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
DB_URL = os.getenv(
    "DB_URL",
    "postgresql://neondb_owner:npg_LoMRfn9gqI0A@ep-noisy-darkness-a1wr0h4z-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

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
DISTRICTS: List[str] = []
WARDS: List[str] = []

@app.on_event("startup")
def _setup():
    # load từ vựng quận/phường cho parser rule-based
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

# ====== PARSER RULES ======
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

def _find_one(cands: List[str], msg: str) -> Optional[str]:
    for c in cands:
        if c and _contains(c, msg):
            return c
    return None

# synonyms -> canonical tags (khớp với DB)
AMENITY_SYNONYMS = {
    "máy lạnh": "air_conditioner", "may lanh": "air_conditioner",
    "máy giặt": "washing_machine", "may giat": "washing_machine",
    "nội thất": "full_furniture", "noi that": "full_furniture",
    "wifi": "wifi"
}
TARGET_SYNONYMS = {
    "sinh viên": "student", "sinh vien": "student",
    "gia đình": "family", "gia dinh": "family",
}
ENV_SYNONYMS = {
    "yên tĩnh": "quiet", "yen tinh": "quiet",
    "an ninh": "secure",
    "gần trường": "near_school", "gan truong": "near_school"
}

def parse_rule_based(message: str) -> dict:
    district = _find_one(DISTRICTS, message)
    ward     = _find_one(WARDS, message)
    pmin, pmax = _parse_price(message)

    amens = []
    for k, canon in AMENITY_SYNONYMS.items():
        if _contains(k, message):
            amens.append(canon)
    targets = [canon for k, canon in TARGET_SYNONYMS.items() if _contains(k, message)]
    envs    = [canon for k, canon in ENV_SYNONYMS.items()    if _contains(k, message)]

    return {
        "province": None,
        "district": district,
        "ward": ward,
        "price_min": pmin,
        "price_max": pmax,
        "area_min": None,
        "area_max": None,
        "amenities": list(set(amens)),
        "targets": list(set(targets)),
        "env_types": list(set(envs)),
        "semantic_query": message
    }

# ====== CORE ======
CANON_AMENITIES = {"pet_friendly", "air_conditioner", "washing_machine", "full_furniture", "wifi"}
CANON_TARGETS = {"student", "family", "female_only", "male_only"}
CANON_ENVS = {"quiet", "secure", "near_school"}

def sanitize_parsed(d: dict) -> dict:
    """Giờ chỉ cần đảm bảo cấu trúc, không cần lọc phức tạp"""
    out = dict(d)
    # Đảm bảo các mảng luôn là list (không phải None)
    out["amenities"] = d.get("amenities") or []
    out["targets"] = d.get("targets") or []
    out["env_types"] = d.get("env_types") or []
    out["keywords"] = d.get("keywords") or []
    return out


def parse_core(message: str) -> dict:
    # Rule-based trước (chỉ địa lý và giá)
    district = _find_one(DISTRICTS, message)
    ward = _find_one(WARDS, message)
    pmin, pmax = _parse_price(message)

    # Nếu có thông tin địa lý hoặc giá, dùng luôn
    if district or ward or (pmin and pmax):
        return {
            "province": None,
            "district": district,
            "ward": ward,
            "price_min": pmin,
            "price_max": pmax,
            "area_min": None,
            "area_max": None,
            "amenities": [],
            "targets": [],
            "env_types": [],
            "keywords": [message],
            "semantic_query": message
        }

    # Fallback LLM cho các trường hợp phức tạp
    system = (
        "Bạn là một trợ lý chuyên trích xuất thông tin từ câu nói của người dùng về nhu cầu tìm nhà trọ. "
        "Nhiệm vụ DUY NHẤT của bạn là trả về một JSON object với các khóa: "
        "province, district, ward, price_min, price_max, area_min, area_max, "
        "amenities, targets, env_types, keywords, semantic_query.\n\n"
        "QUY TẮC: Chỉ trích xuất thông tin CÓ TRONG CÂU. Đối với giá cả, nếu người dùng nói 'tầm/khoảng' ~X triệu, hãy suy ra một khoảng giá hợp lý."
    )
    prompt = f'Người dùng: "{message}"'
    try:
        txt = ollama_generate(PARSE_MODEL, system, prompt, as_json=True, timeout=40)
        raw = json.loads(txt)
        return sanitize_parsed(raw)
    except Exception:
        # Fallback cuối cùng
        return {
            "province": None,
            "district": None,
            "ward": None,
            "price_min": None,
            "price_max": None,
            "area_min": None,
            "area_max": None,
            "amenities": [],
            "targets": [],
            "env_types": [],
            "keywords": [message],
            "semantic_query": message
        }

def search_core(p: Search) -> List[dict]:
    # CHỈ lấy bản ghi hiển thị
    where = ["COALESCE(bz.is_visible, false) = false"]
    args: List = []

    # ILIKE không dấu/không phân biệt hoa-thường dựa trên hàm IMMUTABLE f_unaccent(text)
    # (xem SQL setup ở cuối file này)
    def like_u(expr: str) -> str:
        return f"f_unaccent({expr}) ILIKE f_unaccent(%s)"

    # Tìm theo địa lý với fallback qua name/description/address + FTS
    if p.province:
        where.append(like_u("bz.address"))
        args.append(f"%{p.province}%")

    if p.district:
        where.append("(" + " OR ".join([
            like_u("bz.district"),
            like_u("bz.name"),
            like_u("bz.description"),
            like_u("bz.address"),
            """to_tsvector('simple', f_unaccent(
                   coalesce(bz.name,'')||' '||coalesce(bz.description,'')||' '||
                   coalesce(bz.address,'')||' '||coalesce(bz.district,'')||' '||coalesce(bz.ward,'')
               )) @@ plainto_tsquery('simple', f_unaccent(%s))"""
        ]) + ")")
        args += [f"%{p.district}%"] * 4 + [p.district]

    if p.ward:
        where.append("(" + " OR ".join([
            like_u("bz.ward"),
            like_u("bz.name"),
            like_u("bz.description"),
            like_u("bz.address"),
            """to_tsvector('simple', f_unaccent(
                   coalesce(bz.name,'')||' '||coalesce(bz.description,'')||' '||
                   coalesce(bz.address,'')||' '||coalesce(bz.district,'')||' '||coalesce(bz.ward,'')
               )) @@ plainto_tsquery('simple', f_unaccent(%s))"""
        ]) + ")")
        args += [f"%{p.ward}%"] * 4 + [p.ward]

    # Numeric filters
    if p.price_min is not None: where.append("bz.expected_price >= %s"); args.append(p.price_min)
    if p.price_max is not None: where.append("bz.expected_price <= %s"); args.append(p.price_max)
    if p.area_min  is not None: where.append("bz.area >= %s"); args.append(p.area_min)
    if p.area_max  is not None: where.append("bz.area <= %s"); args.append(p.area_max)


    def build_name_desc_condition(keywords: list[str]) -> tuple[str, list[str]]:
        # tạo "(name ILIKE ? OR description ILIKE ?)" cho mỗi keyword
        conds, args_local = [], []
        for kw in keywords:
            conds.append("(" + like_u("bz.name") + " OR " + like_u("bz.description") + ")")
            args_local += [f"%{kw}%"] * 2
        return " OR ".join(conds), args_local

    # --- Amenities ---
    if p.amenities:
        where.append("""EXISTS (
                    SELECT 1 FROM boarding_zone_amenities a
                    WHERE a.boarding_zone_id = bz.id AND a.amenity_name = ANY(%s)
                )""")
        args.append(p.amenities)

    # --- Targets ---
    if p.targets:
        where.append("""EXISTS (
            SELECT 1 FROM boarding_zone_targets a
            WHERE a.boarding_zone_id = bz.id AND a.target_group = ANY(%s)
        )""")
        args.append(p.targets)

    # --- Env types ---
    if p.env_types:
        where.append("""EXISTS (
            SELECT 1 FROM boarding_zone_environment a
            WHERE a.boarding_zone_id = bz.id AND a.environment_type = ANY(%s)
        )""")
        args.append(p.env_types)

    # --- Name / Description keywords ---
    if p.keywords:
        fts_sql = """to_tsvector('simple', f_unaccent(
                       coalesce(bz.name,'')||' '||coalesce(bz.description,'')
                     )) @@ plainto_tsquery('simple', f_unaccent(%s))"""
        fts_arg = " ".join(p.keywords)

        ilike_conds, ilike_args = [], []
        for kw in p.keywords:
            ilike_conds.append("(" + like_u("bz.name") + " OR " + like_u("bz.description") + ")")
            ilike_args += [f"%{kw}%"] * 2
        ilike_sql = " OR ".join(ilike_conds)

        # Gộp FTS OR ILIKE trong MỘT where-item để làm fallback mềm
        where.append(f"(({fts_sql}) OR ({ilike_sql}))")
        args += [fts_arg] + ilike_args



    base = f"""
      SELECT bz.id, bz.name, bz.expected_price, bz.area, bz.province, bz.district, bz.ward,
             bz.address, bz.description, bz.created_at
      FROM boarding_zones bz
      WHERE {' AND '.join(where)}
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            # Hybrid rank: FTS + embedding nếu có semantic_query
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
                ORDER BY (COALESCE(fts_score,0)*0.4 + COALESCE(emb_score,0)*0.6) DESC,
                         created_at DESC
                LIMIT %s OFFSET %s
                """
                cur.execute(sql, args + [p.semantic_query, qvec, p.limit, (p.page - 1) * p.limit])
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
    # mở rộng ±15% khi min==max
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
