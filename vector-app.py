import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# ========================
# API KEY (ISI PUNYA KAMU)
# ========================
PINECONE_API_KEY = "ISI_API_KEY_KAMU"
OPENAI_API_KEY = "ISI_API_KEY_KAMU"

pc = Pinecone(api_key=PINECONE_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

index_name = "artikel-index"

# ========================
# BUAT INDEX (JIKA BELUM ADA)
# ========================
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# ========================
# DATA ARTIKEL
# ========================
artikel = [
    ("1", "Perkembangan AI Kecerdasan buatan berkembang pesat dalam berbagai bidang industri."),
    ("2", "Belajar Python Python adalah bahasa pemrograman yang mudah dipelajari."),
    ("3", "Kesehatan Jantung Menjaga pola makan penting untuk kesehatan jantung."),
    ("4", "Olahraga Pagi Olahraga di pagi hari membantu meningkatkan energi."),
    ("5", "Blockchain Blockchain digunakan untuk keamanan data."),
    ("6", "E-learning Pendidikan online memudahkan belajar."),
    ("7", "Diet Sehat Makanan bergizi penting untuk tubuh."),
    ("8", "Sepak Bola Olahraga paling populer di dunia."),
    ("9", "Cloud Computing Penyimpanan data online."),
    ("10", "Machine Learning Komputer belajar dari data."),
    ("11", "Air Putih Penting untuk hidrasi tubuh."),
    ("12", "Basket Strategi permainan berkembang."),
    ("13", "Cyber Security Melindungi data pribadi."),
    ("14", "SQL Database relasional."),
    ("15", "Yoga Mengurangi stres."),
    ("16", "Big Data Analisis data besar."),
    ("17", "Renang Melatih seluruh tubuh."),
    ("18", "IoT Perangkat terhubung internet."),
    ("19", "Java Bahasa enterprise."),
    ("20", "Mental Health Penting untuk kesejahteraan.")
]

# ========================
# EMBEDDING FUNCTION
# ========================
def get_embedding(text):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding

# ========================
# INSERT DATA (HANYA SEKALI)
# ========================
if "uploaded" not in st.session_state:
    vectors = []
    for id, text in artikel:
        emb = get_embedding(text)
        vectors.append({
            "id": id,
            "values": emb,
            "metadata": {
                "text": text
            }
        })
    
    index.upsert(vectors=vectors)
    st.session_state.uploaded = True

# ========================
# UI STREAMLIT
# ========================
st.title("🔍 Pencarian Artikel (Vector Database)")

query = st.text_input("Masukkan kata kunci atau kalimat:")

if st.button("Cari"):
    if query:
        query_emb = get_embedding(query)

        result = index.query(
            vector=query_emb,
            top_k=5,
            include_values=False
        )

        st.subheader("Hasil Pencarian:")

        for match in result.matches:
            idx = int(match.id) - 1
            st.write(f"📄 {artikel[idx][1]}")
            st.write(f"Similarity Score: {match.score:.4f}")
            st.write("---")