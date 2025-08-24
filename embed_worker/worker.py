# embed_worker/worker.py
import sys
import psycopg
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer

DB_URL = "postgresql://neondb_owner:npg_LoMRfn9gqI0A@ep-noisy-darkness-a1wr0h4z-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"  # <-- đổi lại của bạn
EMB_MODEL = "intfloat/multilingual-e5-small"  # 384 chiều
BATCH = 1000  # nhúng theo lô để đỡ tốn RAM

def make_text(r: dict) -> str:
    name = r.get("name") or ""
    desc = r.get("description") or ""
    amns = ", ".join(r.get("amenities") or [])
    tgts = ", ".join(r.get("targets") or [])
    envs = ", ".join(r.get("environment") or [])
    return f"""{name}
{desc}
Tiện nghi: {amns}
Đối tượng: {tgts}
Môi trường: {envs}"""

def main():
    print("➡️  Connecting DB…")
    with psycopg.connect(DB_URL) as conn, conn.cursor(row_factory=dict_row) as cur:
        # Thống kê nhanh
        cur.execute("SELECT count(*) AS total, sum((is_visible = true)::int) AS visible FROM boarding_zones;")
        stats = cur.fetchone()
        print(f"📊 total: {stats['total']} | visible: {stats['visible']}")

        # Lấy các bản ghi cần embed (đang hiển thị)
        cur.execute("""
            SELECT bz.id, bz.name, bz.description,
                   COALESCE(array_agg(DISTINCT a.amenity_name)
                            FILTER (WHERE a.amenity_name IS NOT NULL), '{}') AS amenities,
                   COALESCE(array_agg(DISTINCT t.target_group)
                            FILTER (WHERE t.target_group IS NOT NULL), '{}') AS targets,
                   COALESCE(array_agg(DISTINCT e.environment_type)
                            FILTER (WHERE e.environment_type IS NOT NULL), '{}') AS environment
            FROM boarding_zones bz
            LEFT JOIN boarding_zone_amenities  a ON a.boarding_zone_id  = bz.id
            LEFT JOIN boarding_zone_targets    t ON t.boarding_zone_id  = bz.id
            LEFT JOIN boarding_zone_environment e ON e.boarding_zone_id = bz.id
            WHERE bz.is_visible = false
            GROUP BY bz.id
        """)
        rows = cur.fetchall()
        print(f"🧾 Rows to embed: {len(rows)}")
        if not rows:
            print("⚠️  Không có bản ghi nào đủ điều kiện (có thể do is_visible=false hoặc query đang trỏ sai DB).")
            return

        model = SentenceTransformer(EMB_MODEL)

        # Chia batch để embed + update
        total = 0
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i+BATCH]
            texts = [make_text(r) for r in chunk]
            embs  = model.encode(texts, normalize_embeddings=True)

            # cập nhật
            with conn.cursor() as cur2:
                for r, e in zip(chunk, embs):
                    cur2.execute(
                        "UPDATE boarding_zones SET embedding = %s WHERE id = %s",
                        (e.tolist(), r["id"])
                    )
                conn.commit()
            total += len(chunk)
            print(f"✅ Embedded {total}/{len(rows)}")

    print("🎉 DONE")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ ERROR:", e)
        sys.exit(1)
