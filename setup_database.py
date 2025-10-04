import sqlite3
import numpy as np

# connect (creates file if not exists)
conn = sqlite3.connect("faces.db")
c = conn.cursor()

# create table (only once)
c.execute("""
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    embedding BLOB NOT NULL
)
""")
conn.commit()

def register_face_db(name, embedding):
    emb_blob = embedding.astype(np.float32).tobytes()
    c.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, emb_blob))
    conn.commit()
    print(f"[INFO] Registered {name} in database.")

# Load all embeddings into dictionary {name: emb}
def load_faces_db():
    c.execute("SELECT name, embedding FROM faces")
    rows = c.fetchall()
    db = {}
    for name, emb_blob in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        db[name] = emb
    return db

