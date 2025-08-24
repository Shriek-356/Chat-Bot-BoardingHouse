from fastapi import FastAPI
from pydantic import BaseModel
import psycopg, requests, json

DB = "postgresql://user:pass@localhost:5432/boarding"
EMBED = "http://localhost:8001/embed"
OLLAMA = "http://localhost:11434/api/generate"  # qwen2.5:7b-instruct-q4_0

app = FastAPI()

class Search(BaseModel):
    province: str|None=None
    district: str|None=None
    ward: str|None=None
    price_min: float|None=None
    price_max: float|None=None
    area_min:  float|None=None
    area_max:  float|None=None
    amenities: list[str]|None=None        # ["nội thất","giữ xe",...]
    targets:   list[str]|None=None        # ["sinh viên",...]
    env_types: list[str]|None=None        # ["yên tĩnh","hẻm xe hơi",...]
    limit: int = 10
    page:  int = 1
    semantic_query: str|None=None

def embed(text:str)->list[float]:
    return requests.post(EMBED, json={"text": text}).json()["embedding"]

@app.post("/search")
def search(p: Search):
    where = ["bz.is_visible = true"]
    args = []
    if p.province: where.append("bz.province = %s"); args.append(p.province)
    if p.district: where.append("bz.district = %s"); args.append(p.district)
    if p.ward:     where.append("bz.ward = %s");     args.append(p.ward)
    if p.price_min is not None: where.append("bz.expected_price >= %s"); args.append(p.price_min)
    if p.price_max is not None: where.append("bz.expected_price <= %s"); args.append(p.price_max)
    if p.area_min  is not None: where.append("bz.area >= %s");           args.append(p.area_min)
    if p.area_max  is not None: where.append("bz.area <= %s");           args.append(p.area_max)
    if p.amenities:
        where.append("""
          EXISTS (SELECT 1 FROM boarding_zone_amenities a
                  WHERE a.boarding_zone_id = bz.id AND a.amenity_name = ANY(%s))
        """); args.append(p.amenities)
    if p.targets:
        where.append("""
          EXISTS (SELECT 1 FROM boarding_zone_targets t
                  WHERE t.boarding_zone_id = bz.id AND t.target_group = ANY(%s))
        """); args.append(p.targets)
    if p.env_types:
        where.append("""
          EXISTS (SELECT 1 FROM boarding_zone_environment e
                  WHERE e.boarding_zone_id = bz.id AND e.environment_type = ANY(%s))
        """); args.append(p.env_types)

    base = f"""
      SELECT bz.id, bz.name, bz.expected_price, bz.area, bz.province, bz.district, bz.ward,
             bz.address, bz.description, bz.embedding, bz.created_at
      FROM boarding_zones bz
      WHERE {' AND '.join(where)}
    """

    if p.semantic_query:
        qvec = embed(p.semantic_query)
        sql = f"""
        WITH filtered AS ({base}),
        vec AS (
          SELECT id, 1 - (embedding <=> %s) AS semantic_score
          FROM filtered
          ORDER BY embedding <=> %s
          LIMIT 200
        )
        SELECT f.*, v.semantic_score
        FROM filtered f
        JOIN vec v USING (id)
        ORDER BY v.semantic_score DESC, f.created_at DESC
        LIMIT %s OFFSET %s
        """
        qargs = args + [qvec, qvec, p.limit, (p.page-1)*p.limit]
    else:
        sql = f"{base} ORDER BY created_at DESC LIMIT %s OFFSET %s"
        qargs = args + [p.limit, (p.page-1)*p.limit]

    with psycopg.connect(DB) as conn:
        cur = conn.cursor()
        cur.execute(sql, qargs)
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return {"items": rows}

@app.post("/parse")
def parse(body: dict):
    msg = body.get("message","")
    prompt = f"""
Bạn là bộ trích xuất JSON. Người dùng: \"{msg}\"
Trả về JSON DUY NHẤT: {{
 "province":string|null, "district":string|null, "ward":string|null,
 "price_min":number|null, "price_max":number|null,
 "area_min":number|null, "area_max":number|null,
 "amenities":string[], "targets":string[], "env_types":string[],
 "semantic_query":string
}}
Không thêm chữ nào ngoài JSON.
"""
    resp = requests.post(OLLAMA, json={"model":"qwen2.5:7b-instruct-q4_0","prompt":prompt,"stream":False}).json().get("response","{}")
    s = resp.strip()
    s = s[s.find("{"): s.rfind("}")+1]
    try: obj = json.loads(s)
    except: obj = {"province":None,"district":None,"ward":None,"price_min":None,"price_max":None,"area_min":None,"area_max":None,"amenities":[],"targets":[],"env_types":[],"semantic_query":msg}
    return obj

@app.post("/chat")
def chat(body: dict):
    msg = body.get("message","")
    parsed = requests.post("http://localhost:8000/parse", json={"message": msg}).json()

    sr = requests.post("http://localhost:8000/search", json={
        "province": parsed.get("province"),
        "district": parsed.get("district"),
        "ward":     parsed.get("ward"),
        "price_min":parsed.get("price_min"),
        "price_max":parsed.get("price_max"),
        "area_min": parsed.get("area_min"),
        "area_max": parsed.get("area_max"),
        "amenities":parsed.get("amenities") or [],
        "targets":  parsed.get("targets")   or [],
        "env_types":parsed.get("env_types") or [],
        "semantic_query": parsed.get("semantic_query") or msg,
        "limit": 5
    }).json()["items"]

    ctx = []
    for r in sr:
        desc = (r.get("description") or "")[:160]
        ctx.append(f"- {r['name']} | {r.get('expected_price')}đ | {r.get('district','')} {r.get('ward','')} | {r.get('address','')}\n  {desc}...")
    context = "\n".join(ctx) or "(không có kết quả phù hợp)"

    prompt = f"""Bạn là chatbot tư vấn trọ. Yêu cầu: {msg}
Kết quả:
{context}

Hãy tư vấn 3–6 câu, nêu lý do phù hợp (giá/khu vực/tiện nghi) và gợi ý 1–2 lọc thêm. Không bịa."""
    ans = requests.post(OLLAMA, json={"model":"qwen2.5:7b-instruct-q4_0","prompt":prompt,"stream":False}).json().get("response","")
    return {"answer": ans, "items": sr}