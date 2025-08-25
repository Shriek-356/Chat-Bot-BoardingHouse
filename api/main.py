import os, time, json, re, unicodedata, math
from typing import List, Optional
from functools import lru_cache

import requests
from requests.exceptions import RequestException, Timeout
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from psycopg_pool import ConnectionPool
from sentence_transformers import SentenceTransformer

# =========================
# ====== CONFIG ===========
# =========================
DB_URL = os.getenv(
    "DB_URL",
    "postgresql://neondb_owner:npg_LoMRfn9gqI0A@ep-noisy-darkness-a1wr0h4z-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# Cho phép chọn 2 model khác nhau bằng biến môi trường
PARSE_MODEL = os.getenv("PARSE_MODEL", "qwen2.5:3b-instruct-q4_0")
CHAT_MODEL  = os.getenv("CHAT_MODEL",  "qwen2.5:3b-instruct-q4_0")

# Gen opts
GEN_PARSE_OPTS = {"temperature": 0.0,  "num_predict": 120, "keep_alive": "30m", "num_thread": os.cpu_count() or 4}
GEN_CHAT_OPTS  = {"temperature": 0.25, "num_predict": 120, "keep_alive": "30m", "num_thread": os.cpu_count() or 4}

# Embedding model (384 chiều)
_embed_model = SentenceTransformer("intfloat/multilingual-e5-small")

# DB pool
pool = ConnectionPool(conninfo=DB_URL, min_size=1, max_size=10)

app = FastAPI(title="Boarding Chatbot API")

# =========================
# ====== MODELS ===========
# =========================
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

# =========================
# ====== STARTUP ==========
# =========================
DISTRICTS: List[str] = []
WARDS: List[str] = []

@app.on_event("startup")
def _setup():
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # extensions cần thiết
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS unaccent;")
                except Exception as e:
                    print("[WARN] cannot enable unaccent:", e)

                # nạp vocab quận/phường để match rule-based
                cur.execute("SELECT DISTINCT district FROM boarding_zones WHERE district IS NOT NULL LIMIT 5000")
                globals()["DISTRICTS"] = [r[0] for r in cur.fetchall()]
                cur.execute("SELECT DISTINCT ward FROM boarding_zones WHERE ward IS NOT NULL LIMIT 10000")
                globals()["WARDS"] = [r[0] for r in cur.fetchall()]
        print(f"[VOCAB] districts={len(DISTRICTS)} wards={len(WARDS)}")
        print(f"[LLM] PARSE={PARSE_MODEL} | CHAT={CHAT_MODEL}")
    except Exception as e:
        print("[WARN] startup:", e)

# =========================
# ====== UTILS ============
# =========================
def timing(label: str, t0: float) -> float:
    t1 = time.perf_counter()
    print(f"[TIMING] {label}: {t1 - t0:.2f}s")
    return t1

def _safe_json_loads(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        # cố tách phần JSON khi model lỡ in thêm text
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {}

def ollama_generate(model: str, system: str, prompt: str, *, as_json=False, timeout=60, retry=1) -> str:
    opts = GEN_PARSE_OPTS if as_json else GEN_CHAT_OPTS
    payload = {"model": model, "system": system or "", "prompt": prompt, "options": opts, "stream": False}
    if as_json:
        payload["format"] = "json"
    last_err = None
    for i in range(retry + 1):
        try:
            t0 = time.perf_counter()
            r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            print(f"[OLLAMA] {model} {r.status_code} in {time.perf_counter()-t0:.2f}s (try {i+1})")
            r.raise_for_status()
            return r.json().get("response", "")
        except (RequestException, Timeout) as e:
            last_err = e
            time.sleep(0.5 * (i + 1))
    raise last_err

@lru_cache(maxsize=512)
def embed_vec_literal(text: str) -> str:
    vec = _embed_model.encode([text], normalize_embeddings=True)[0].tolist()
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

def widen_price(pmin: Optional[float], pmax: Optional[float]):
    if pmin is not None and pmax is not None and abs(pmin - pmax) < 1e-6:
        delta = max(500_000, int(pmin * 0.15))
        return max(0, pmin - delta), pmax + delta
    return pmin, pmax

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", _strip_accents(s.lower())).strip()

def _aliases_for_area(name: str) -> List[str]:
    s = _norm(name)
    base = re.sub(r"^(quan|huyen|thi xa|tp|thanh pho|phuong|xa)\s+", "", s)
    return [s, base]

# =========================
# == RULE-BASED PARSER ====
# =========================
def _parse_price(msg: str):
    txt = _norm(msg)

    # range: 3-4 tr / 3–4 triệu / 3 to 4 tr
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*[-–to]+\s*(\d+(?:[.,]\d+)?)\s*(tr|trieu|million)', txt)
    if m:
        a = float(m.group(1).replace(",", ".")) * 1_000_000
        b = float(m.group(2).replace(",", ".")) * 1_000_000
        return (min(a,b), max(a,b))

    # "duoi/<=/< 3 tr"
    m = re.search(r'(duoi|<|<=|khong vuot|toi da)\s*(\d+(?:[.,]\d+)?)\s*(tr|trieu|million)', txt)
    if m:
        v = float(m.group(2).replace(",", ".")) * 1_000_000
        return (0, v)

    # "tren/>=/> 2 tr"
    m = re.search(r'(tren|>|>=|toi thieu)\s*(\d+(?:[.,]\d+)?)\s*(tr|trieu|million)', txt)
    if m:
        v = float(m.group(2).replace(",", ".")) * 1_000_000
        return (v, None)

    # 2m5 (=2.5tr)
    m = re.search(r'(\d+)\s*m\s*(\d)', txt)
    if m:
        v = float(m.group(1) + "." + m.group(2)) * 1_000_000
        return max(0, v*0.85), v*1.15

    # 2.5tr / 2,5 tr
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*(tr|trieu|million)', txt)
    if m:
        v = float(m.group(1).replace(",", ".")) * 1_000_000
        return max(0, v*0.85), v*1.15

    # 500k / 500 nghìn
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*(k|nghin|nghìn)', txt)
    if m:
        v = float(m.group(1).replace(",", ".")) * 1_000
        return max(0, v*0.85), v*1.15

    # số dài: 3 000 000 / 3000000 / 2.300.000
    m = re.search(r'(\d[\d .]{5,})\s*(?:d|vnd)?', txt)
    if m:
        v = float(re.sub(r"[ .]", "", m.group(1)))
        return max(0, v*0.85), v*1.15

    return None, None

def _find_area_longest(cands: List[str], msg: str) -> Optional[str]:
    nm = _norm(msg)
    best = None
    best_len = -1
    for c in cands:
        for a in _aliases_for_area(c):
            if not a:
                continue
            if a in nm or nm in a:
                core = re.sub(r"^(quan|huyen|thi xa|tp|thanh pho|phuong|xa)\s+", "", a).strip()
                if len(core) > best_len:
                    best, best_len = core, len(core)
    return best

def parse_rule_based(message: str) -> dict:
    district = _find_area_longest(DISTRICTS, message)
    ward     = _find_area_longest(WARDS, message)
    pmin, pmax = _parse_price(message)

    amens = []
    amen_aliases = {
        "nội thất": ["noi that", "full nt", "full noi that", "full do", "co noi that"],
        "máy lạnh": ["may lanh", "dieu hoa", "aircon", "ac"],
        "máy giặt": ["may giat", "giat chung", "giat rieng"],
        "wifi": ["internet", "mang", "wifi"]
    }
    nm = _norm(message)
    for canon, keys in amen_aliases.items():
        for k in [canon] + keys:
            if _norm(k) in nm:
                amens.append(canon); break

    return {
        "province": None,
        "district": district,
        "ward": ward,
        "price_min": pmin,
        "price_max": pmax,
        "area_min": None,
        "area_max": None,
        "amenities": sorted(list(set(amens))),
        "targets": [],
        "env_types": [],
        "semantic_query": message if message and len(message) > 2 else None
    }

# =========================
# ====== CORE =============
# =========================
def parse_core(message: str) -> dict:
    rb = parse_rule_based(message)
    # Nếu đã có quận/phường/giá/amenities rồi thì dùng luôn rule-based cho nhanh & chắc
    if rb.get("district") or rb.get("ward") or (rb.get("price_min") and rb.get("price_max")) or rb["amenities"]:
        return rb

    system = (
        "Bạn chỉ trích xuất JSON cho truy vấn phòng trọ Việt Nam. "
        "Chỉ trả JSON với khóa: province,district,ward,price_min,price_max,area_min,area_max,amenities,targets,env_types,semantic_query."
    )
    prompt = f'Người dùng: "{message}"\nNếu thấy "tầm/khoảng ~X triệu", suy ra ±15%. Nếu không chắc, để null.'
    try:
        txt = ollama_generate(PARSE_MODEL, system, prompt, as_json=True, timeout=40, retry=1)
        data = _safe_json_loads(txt)
        return data or rb
    except Exception:
        return rb

def search_core(p: Search) -> List[dict]:
    # Sửa thành true để chỉ lấy bài hiển thị
    where = ["COALESCE(bz.is_visible, false) = true"]
    args: List = []

    def like_unaccent(col: str) -> str:
        # so sánh lower(unaccent(col)) với lower(unaccent(%s))
        return f"lower(unaccent({col})) LIKE lower(unaccent(%s))"

    if p.province:
        where.append(like_unaccent("bz.province")); args.append(f"%{_norm(p.province)}%")

    if p.district:
        d = _norm(p.district)
        where.append("(" + " OR ".join([like_unaccent("bz.district")]*3) + ")")
        args += [f"%{d}%", f"%quan {d}%", f"%huyen {d}%"]

    if p.ward:
        w = _norm(p.ward)
        where.append("(" + " OR ".join([like_unaccent("bz.ward")]*3) + ")")
        args += [f"%{w}%", f"%phuong {w}%", f"%xa {w}%"]

    if p.price_min is not None: where.append("bz.expected_price >= %s"); args.append(p.price_min)
    if p.price_max is not None: where.append("bz.expected_price <= %s"); args.append(p.price_max)
    if p.area_min  is not None: where.append("bz.area >= %s"); args.append(p.area_min)
    if p.area_max  is not None: where.append("bz.area <= %s"); args.append(p.area_max)

    if p.amenities:
        where.append("""EXISTS (
            SELECT 1 FROM boarding_zone_amenities a
            WHERE a.boarding_zone_id = bz.id
              AND lower(a.amenity_name) = ANY(%s)
        )""")
        args.append([s.lower() for s in p.amenities])

    if p.targets:
        where.append("""EXISTS (
            SELECT 1 FROM boarding_zone_targets t
            WHERE t.boarding_zone_id = bz.id
              AND lower(t.target_group) = ANY(%s)
        )""")
        args.append([s.lower() for s in p.targets])

    if p.env_types:
        where.append("""EXISTS (
            SELECT 1 FROM boarding_zone_environment e
            WHERE e.boarding_zone_id = bz.id
              AND lower(e.environment_type) = ANY(%s)
        )""")
        args.append([s.lower() for s in p.env_types])

    base = f"""
      SELECT bz.id, bz.name, bz.expected_price, bz.area, bz.province, bz.district, bz.ward,
             bz.address, bz.description, bz.created_at
      FROM boarding_zones bz
      WHERE {' AND '.join(where)}
    """

    limit = max(1, min(int(p.limit or 10), 50))
    offset = max(0, (int(p.page or 1) - 1) * limit)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            if p.semantic_query and len(p.semantic_query) > 2:
                try:
                    qvec = embed_vec_literal(p.semantic_query)
                    try:
                        cur.execute("SELECT set_config('ivfflat.probes', '8', true)")
                    except Exception as e:
                        print("[WARN] cannot set ivfflat.probes:", e)

                    boost = " + 0.05" if p.district else ""
                    sql = f"""
                    WITH filtered AS ({base}),
                    vec AS (
                      SELECT f.id, 1 - (bz.embedding <=> %s::vector) AS semantic_score
                      FROM filtered f
                      JOIN boarding_zones bz ON bz.id = f.id
                      ORDER BY bz.embedding <=> %s::vector
                      LIMIT 150
                    )
                    SELECT f.id, f.name, f.expected_price, f.area, f.province, f.district, f.ward,
                           f.address, f.description, f.created_at,
                           (v.semantic_score{boost}) AS semantic_score
                    FROM filtered f
                    JOIN vec v USING (id)
                    ORDER BY semantic_score DESC, f.created_at DESC
                    LIMIT %s OFFSET %s
                    """
                    cur.execute(sql, args + [qvec, qvec, limit, offset])
                except Exception as e:
                    print("[WARN] semantic search failed, fallback to filter only:", e)
                    sql = f"{base} ORDER BY created_at DESC LIMIT %s OFFSET %s"
                    cur.execute(sql, args + [limit, offset])
            else:
                sql = f"{base} ORDER BY created_at DESC LIMIT %s OFFSET %s"
                cur.execute(sql, args + [limit, offset])

            cols = [c.name for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

# =========================
# ====== ENDPOINTS ========
# =========================
@app.get("/health")
def health():
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/parse")
def parse(body: dict):
    t0 = time.perf_counter()
    out = parse_core((body.get("message") or "").strip())
    timing("parse", t0)
    return out

@app.post("/search")
def search(p: Search):
    t0 = time.perf_counter()
    try:
        items = search_core(p)
        timing("search", t0)
        return {"items": items}
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

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
            area_min=parsed.get("area_min"), area_max=parsed.get("area_max"),
            amenities=(parsed.get("amenities") or [])[:8],
            targets=(parsed.get("targets") or [])[:8],
            env_types=(parsed.get("env_types") or [])[:8],
            limit=5, page=1,
            semantic_query=parsed.get("semantic_query") or msg,
        )

        items = search_core(params)
        t2 = timing("chat.search", t1)

        # Chuẩn bị context gọn gàng
        lines = []
        for r in items[:8]:
            desc = (r.get("description") or "")
            desc = (desc[:140] + "…") if len(desc) > 150 else desc
            price = r.get('expected_price')
            try:
                price_str = f"{int(price):,}đ" if price is not None else "—"
            except Exception:
                price_str = f"{price}đ" if price is not None else "—"
            lines.append(f"- {r['name']} | {price_str} | {r.get('district','')} {r.get('ward','')}\n  {desc}")

        context = "\n".join(lines) if lines else "(không có kết quả phù hợp)."

        system = (
            "Bạn là chatbot tư vấn trọ tiếng Việt, trả lời gọn, trung thực, không bịa. "
            "Nếu không có kết quả, hãy khuyên người dùng nới điều kiện (khu vực lân cận, khoảng giá ±15–20%, bỏ bớt tiện nghi)."
        )
        prompt = (
            f"Yêu cầu: {msg}\nKết quả:\n{context}\n\n"
            "Viết 3–6 câu tư vấn: nêu vì sao phù hợp (giá/khu vực/tiện nghi), gợi ý 1–3 cách lọc thêm. "
            "Giọng thân thiện, không liệt kê quá dài."
        )

        try:
            answer = ollama_generate(CHAT_MODEL, system, prompt, as_json=False, timeout=90, retry=1)
        except Exception:
            # Fallback: trả lời tay ngắn gọn
            if items:
                answer = "Mình đã lọc theo yêu cầu và gợi ý vài lựa chọn ở trên. Bạn có thể thử nới khoảng giá ±15% hoặc chọn khu lân cận để có thêm kết quả."
            else:
                answer = "Hiện chưa có kết quả khớp. Bạn thử nới khoảng giá (±20%), mở rộng sang quận kế bên, hoặc bỏ bớt một tiện nghi nhé."

        timing("chat.answer", t2)
        timing("chat.total", t_all)
        return {"answer": answer, "items": items, "parsed": parsed}
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
