# api/main.py — fast & tolerant (rule-based parser + FTS + pgvector hybrid rank)
import os, time, json, re, unicodedata
from typing import List, Optional, Dict
from functools import lru_cache

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from psycopg_pool import ConnectionPool
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pathlib import Path

# ====== LOAD ENV ======
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# ====== CONFIG (DB) ======
DB_URL = (
    "postgresql://neondb_owner:npg_LoMRfn9gqI0A@ep-noisy-darkness-a1wr0h4z-pooler.ap-southeast-1.aws.neon.tech/neondb"
    "?sslmode=require"
    "&connect_timeout=15"
    "&hostaddr=13.228.184.177"
    "&keepalives=1&keepalives_idle=30&keepalives_interval=10&keepalives_count=5"
)

# ====== CONFIG (LLM hosted) ======
# Bạn có thể trỏ tới OpenAI/Groq/OpenRouter miễn là endpoint tương thích /v1/chat/completions
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")
LLM_MODEL    = os.getenv("LLM_MODEL", "deepseek/deepseek-chat:free")  # đổi theo nhà cung cấp bạn dùng

def _build_completions_url() -> str:
    base = (os.getenv("LLM_BASE_URL") or "").rstrip("/")
    if not base:
        raise RuntimeError("LLM_BASE_URL is missing")

    host = base.split("//", 1)[-1]
    # OpenRouter: cần '/api/v1'
    if "openrouter.ai" in host:
        if not base.endswith("/api/v1"):
            base = base + "/api/v1"
    # OpenAI: cần '/v1'
    elif "openai.com" in host:
        if not base.endswith("/v1"):
            base = base + "/v1"
    # Groq (OpenAI-compatible, thường dùng '/openai/v1')
    elif "api.groq.com" in host:
        if not base.endswith("/openai/v1") and not base.endswith("/v1"):
            base = base + "/openai/v1"

    return base + "/chat/completions"

def hosted_generate(system: str, prompt: str, temperature=0.25, max_tokens=500, timeout=30) -> str:
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY is missing")

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    # BẮT BUỘC cho OpenRouter:
    if "openrouter.ai" in (os.getenv("LLM_BASE_URL") or ""):
        headers["HTTP-Referer"] = "http://localhost"
        headers["X-Title"] = "Boarding House Chatbot"

    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system or ""},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    url = _build_completions_url()
    r = requests.post(url, headers=headers, json=data, timeout=timeout)
    print(f"[HOSTED] POST {url} -> {r.status_code}  {r.text[:200]}")
    r.raise_for_status()

    j = r.json()
    return (j.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""


# Ngưỡng semantic (0..1). Đặt 0.0 nếu không muốn chặn.
SEM_MIN_SIM = float(os.getenv("SEM_MIN_SIM", "0.15"))

# Embedding model (768 chiều) — chạy local bằng SentenceTransformers
_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("intfloat/multilingual-e5-base")
    return _embed_model

@lru_cache(maxsize=512)
def embed_vec_literal(text: str) -> str:
    vec = _get_embed_model().encode([text], normalize_embeddings=True)[0].tolist()
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

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

class ChatHistory(BaseModel):
    message: str
    history: List[dict] = []

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


@app.get("/__llm_info")
def llm_info():
    from inspect import getsource
    return {
        "LLM_BASE_URL": os.getenv("LLM_BASE_URL"),
        "LLM_MODEL": os.getenv("LLM_MODEL"),
        "HAS_KEY": bool(os.getenv("LLM_API_KEY")),
        "COMPLETIONS_URL": _build_completions_url(),
    }

# ====== UTILS ======
def timing(label: str, t0: float) -> float:
    t1 = time.perf_counter()
    print(f"[TIMING] {label}: {t1 - t0:.2f}s")
    return t1

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

    def _to_vnd(num: str, unit: str = "tr") -> float:
        val = float(num.replace(",", "."))
        if unit in ("k", "nghin", "nghìn"):
            return val * 1_000
        # mặc định triệu
        return val * 1_000_000

    # 1) Khoảng "từ A đến B"
    m = re.search(
        r'tu\s*(\d+(?:[.,]\d+)?)\s*(k|nghin|nghìn|tr|trieu|m|million)?\s*(?:den|-|toi)\s*(\d+(?:[.,]\d+)?)\s*(k|nghin|nghìn|tr|trieu|m|million)?',
        txt
    )
    if m:
        a = _to_vnd(m.group(1), (m.group(2) or "tr"))
        b = _to_vnd(m.group(3), (m.group(4) or "tr"))
        lo, hi = sorted([a, b])
        # nới nhẹ 2% để thân thiện
        return max(0, lo * 0.98), hi * 1.02

    # 2) Một giá trị + toán tử so sánh
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*(k|nghin|nghìn|tr|trieu|m|million)', txt)
    if m:
        v = _to_vnd(m.group(1), m.group(2))
        is_under = any(k in txt for k in ['duoi', '<=', 'toi da', 'max', 'khong qua', '<'])
        is_over  = any(k in txt for k in ['tren', '>=', 'toi thieu', 'min', 'tu ', 'khong duoi', '>'])
        is_around = any(k in txt for k in ['khoang', 'tam', 'xap xi', 'gan', '~', 'cỡ', 'co '])

        if is_under:
            return None, v  # chỉ đặt trần
        if is_over:
            return v, None  # chỉ đặt sàn
        if is_around:
            return max(0, v * 0.85), v * 1.15
        # mặc định: nới nhẹ 10%
        return max(0, v * 0.9), v * 1.1

    # 3) Dạng số dài (VND thô)
    m = re.search(r'(\d[\d .]{5,})\s*(?:d|vnd|vnđ)?', txt)
    if m:
        v = float(re.sub(r"[ .]", "", m.group(1)))
        return max(0, v * 0.9), v * 1.1

    return None, None

def _find_one(cands: List[str], msg: str) -> Optional[str]:
    for c in cands:
        if c and _contains(c, msg):
            return c
    return None

# --- GEO NORMALIZATION ---
GEO_PREFIX_RE = re.compile(
    r'\b('
    r'q\.?|quan|quận|h\.?|huyen|huyện|'
    r'tp|t\.p\.?|thanh pho|thành phố|'
    r'thi xa|tx|thi tran|tt|'    
    r'xa|x\.?|'                   
    r'phuong|p\.?|phường'
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
    """
    - Ưu tiên tên dài (tránh '5' ăn '5tr5')
    - Ứng viên chỉ là số: chỉ nhận khi có tiền tố rõ ràng (quận/phường/xã/tt)
    - Dùng word-boundary khi so khớp
    """
    raw = _strip_accents((msg or "").lower())   # giữ nguyên tiền tố để bắt "xa 5"...
    ms  = _canon_geo(msg)                       # đã bỏ tiền tố, normalize để so cụm chữ

    ordered = sorted(
        (c for c in cands if c),
        key=lambda x: len(_canon_geo(x or "")),
        reverse=True
    )

    for c in ordered:
        cs = _canon_geo(c or "")
        if not cs:
            continue

        # Nếu ứng viên chỉ là số (vd "5"), yêu cầu có tiền tố rõ ràng trong câu gốc
        if cs.isdigit():
            prefix = r'(q\.?|quan|quận|phuong|p\.?|phường|xa|x\.?|thi tran|tt|thi xa|tx)'
            pattern = rf'\b{prefix}\s*0*{cs}\b'
            if re.search(pattern, raw):
                return c
            continue

        # Tên chữ: match theo ranh giới từ
        if re.search(rf'\b{re.escape(cs)}\b', ms):
            return c

    return None

# ====== CORE ======
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
    "có", "nào", "ở", "o", "cho", "không", "khong", "tìm", "tim", "phòng", "phong",
    "trọ", "tro", "cần", "can", "giúp", "giup", "giá", "gia", "khoảng", "khoang",
    "quận", "quan", "huyện", "huyen", "phường", "phuong", "tp", "t p", "thanh", "pho",
    "tôi", "toi", "mình", "minh", "muốn", "muon", "cần tìm", "can tim", "tim phong",
    "phong tro", "tro", "nha", "nhà", "cho thue", "thuê", "thue", "gia re", "rẻ",
    "tot", "tốt", "dep", "đẹp", "san", "sẵn", "nao", "nào", "day", "đầy", "du", "đủ","tầm", "tạm", "dưới", "trên", "tối", "tối đa", "tối thiểu", "từ", "đến", "khoảng", "xấp", "xấp xỉ", "gần", "cỡ", "max", "min"
}

VN_STOPWORDS_NORM = { _strip_accents(w) for w in VN_STOPWORDS }

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

def _extract_regular_keywords(text: str) -> List[str]:
    """Trích keyword có nghĩa; loại bỏ stopwords & token có số."""
    if not text:
        return []
    text = _strip_accents(text.lower())
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    keywords = []
    for w in words:
        if len(w) <= 2:
            continue
        if w in VN_STOPWORDS_NORM:
            continue
        # loại token có chứa số (vd '6tr', '5m')
        if any(ch.isdigit() for ch in w):
            continue
        # loại đơn vị tiền
        if w in {"tr", "trieu", "vnd", "dong", "vnđ"}:
            continue
        if w not in keywords:
            keywords.append(w)

    return keywords[:2]

def _extract_keywords(message: str) -> List[str]:
    kws = _pick_phrases(message, PET_SYNS)
    msg_wo_geo = _remove_geo_from_text(message)
    kws += [k for k in _pick_phrases(msg_wo_geo, AMENITY_PHRASES) if k not in kws]
    regular_kws = _extract_regular_keywords(msg_wo_geo)
    kws.extend([kw for kw in regular_kws if kw not in kws])
    return kws[:5]

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

    # Province strict (+ address fallback)
    if p.province:
        canon = _canon_geo(p.province)
        col_canon = _sql_canon("bz.province")
        addr_canon = _sql_canon("bz.address")
        strict = f"(({col_canon} <> '') AND {col_canon} ILIKE %s)"
        fallback = f"(((bz.province IS NULL OR bz.province = '')) AND {addr_canon} ILIKE %s)"
        where.append(f"({strict} OR {fallback})")
        args += [f"%{canon}%", f"%{canon}%"]

    # District strict (+ address fallback)
    if p.district:
        canon = _canon_geo(p.district)
        col_canon = _sql_canon("bz.district")
        addr_canon = _sql_canon("bz.address")
        strict = f"(({col_canon} <> '') AND {col_canon} ILIKE %s)"
        fallback = f"(((bz.district IS NULL OR bz.district = '')) AND {addr_canon} ILIKE %s)"
        where.append(f"({strict} OR {fallback})")
        args += [f"%{canon}%", f"%{canon}%"]

    # Ward strict (+ address fallback)
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

    # Keywords (AND)
    if p.keywords:
        for kw in p.keywords:
            where.append("""
                to_tsvector('simple', f_unaccent(
                    coalesce(bz.name,'')||' '||coalesce(bz.description,'')||' '||
                    coalesce(bz.address,'')||' '||coalesce(bz.district,'')||' '||
                    coalesce(bz.ward,'')||' '||coalesce(bz.province,'')
                )) @@ websearch_to_tsquery('simple', f_unaccent(%s))
            """)
            args.append(kw)

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

system = """Bạn là trợ lý tìm phòng trọ tiếng Việt. Luôn:
- Suy nghĩ nội bộ rồi chỉ đưa ra KẾT QUẢ CUỐI cùng (không kể bước suy luận).
- Trả lời rõ ràng, lịch sự, súc tích; dùng bullet, tiêu đề, emoji phù hợp.
- Nếu kết quả ít hoặc không có, đề xuất cách nới điều kiện cụ thể (giá, quận lân cận, tiện nghi)."""

def _build_chat_prompt(msg: str, items: List[dict], context: str) -> str:
    # context = danh sách phòng dạng "1. Tên | Giá | Quận, Phường\n   Mô tả…"
    return f"""
YÊU CẦU NGƯỜI DÙNG: {msg}

KẾT QUẢ TÌM THẤY (tối đa 6 phòng):
{context}

NHIỆM VỤ:
1) Chấm điểm từng phòng theo thang 0..100 dựa trên:
   - Khớp địa lý (tỉnh/quận/phường)
   - Khớp giá (ưu tiên gần giá target)
   - Diện tích/tiện nghi nổi bật
2) Chọn 1-2 phòng nổi bật nhất và giải thích ngắn gọn vì sao.
3) Đưa 3 gợi ý cụ thể để mở rộng hoặc tinh chỉnh tìm kiếm.

ĐỊNH DẠNG TRẢ LỜI:
- Chào + tóm tắt 1 dòng
- **Phòng nổi bật** (1–2): tên, giá, khu vực, 1–2 ưu điểm
- **Lựa chọn khác** (nếu có): liệt kê ngắn
- **Gợi ý chỉnh tiêu chí**: 2–3 gạch đầu dòng
"""

def _generate_suggestions(items: List[dict], parsed: dict) -> List[str]:
    suggestions = []
    if not items:
        if parsed.get('price_min'):
            suggestions.append("Thử mở rộng khoảng giá ±20%")
        if parsed.get('district'):
            suggestions.append("Tìm các quận lân cận")
        suggestions.append("Giảm bớt yêu cầu về tiện nghi")
    elif len(items) < 3:
        suggestions.append("Tìm thêm phòng với tiêu chí tương tự")
    return suggestions[:3]


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
    chat_history = body.get("history", [])

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
            limit=8, page=1,
            keywords=parsed.get("keywords") or [],
            semantic_query=parsed.get("semantic_query") or msg,
        )
        items = search_core(params)
        if not items and (params.keywords or parsed.get("keywords")):
            params = params.copy(update={"keywords": []})
            items = search_core(params)

        t2 = timing("chat.search", t1)

        # Build context
        lines = []
        for i, r in enumerate(items[:6]):
            price = f"{r.get('expected_price', 0):,.0f}đ".replace(",", ".")
            district_ward = f"{r.get('district', '')} {r.get('ward', '')}".strip()
            desc = (r.get('description') or '')[:280]
            lines.append(f"{i + 1}. {r['name']} | {price} | {district_ward}\n   {desc}...")
        context = "\n\n".join(lines) if lines else "Không tìm thấy phòng nào phù hợp."
        prompt = _build_chat_prompt(msg, items, context)

        if chat_history:
            history_context = "\n".join([f"User: {h.get('user', '')}\nBot: {h.get('bot', '')}"
                                         for h in chat_history[-3:]])
            prompt = f"""LỊCH SỬ CHAT:
{history_context}

HIỆN TẠI:
{prompt}"""

        # === Hosted only ===
        answer = hosted_generate(system, prompt, temperature=0.2, max_tokens=500, timeout=45)
        if not answer:
            answer = (
                "Hiện mình chưa soạn được tư vấn tự nhiên từ mô hình, "
                "nhưng dưới đây là các phòng phù hợp bạn có thể tham khảo. "
                "Bạn thử nới khoảng giá (±20%) hoặc mở rộng sang quận lân cận để có thêm kết quả nhé."
            )

        timing("chat.answer", t2)
        timing("chat.total", t_all)

        return {
            "answer": answer,
            "items": items[:5],
            "suggestions": _generate_suggestions(items, parsed)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/v2")
def chat_v2(chat_data: ChatHistory):
    """Phiên bản chat hỗ trợ history"""
    return chat({"message": chat_data.message, "history": chat_data.history})
