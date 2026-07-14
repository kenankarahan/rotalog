import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

df = pd.read_csv("simulated_data.csv")

def process_data(row):
    departure = row["Kalkış"]
    destination = row["Hedef"]
    distance = row["Mesafe (KM)"]
    weather = row["Hava Şartları"]
    traffic = row["Trafik Yoğunluğu"]
    check = row["Araç Kontrolü"]
    driving_time = row["Ortalama Günlük Sürüş Saati"]
    automobile_type = row["Araç Tipi"]
    risk_score = row["Risk Skoru"]
    status = row["Durum"]
    co2_emission = row["CO2 Emisyonu (KG)"]

    processed_data = (f"Kalkış noktası {departure}, hedefi {destination} olan ve {distance} KM mesafe kat eden operasyonda kullanılan araç tipi {automobile_type} olarak kaydedilmiştir. "
                      f"hava şartları {weather}, trafik yoğunluğu {traffic} olarak kaydedilmiştir. "
                      f"Yola çıkmadan önce araç kontrolü {check}. Sürücünün toplam sürüş saati {driving_time} saattir. "
                      f"Bu operasyonun risk skoru {risk_score:.2f} olup, süreç {status} durumla sonuçlanmıştır. "
                      f"Harcanan CO2 emisyonu {co2_emission:.2f} KG'dır.")

    return processed_data

df["Processed_Data"] = df.apply(process_data, axis=1)

print(df["Processed_Data"].head(2))

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

chroma_client = chromadb.PersistentClient(path="./logistic_vectoral_db")

collection = chroma_client.get_or_create_collection(name="logistic_operations", metadata={"hnsw:space": "cosine"})

documents = df["Processed_Data"].tolist()
op_ids = [f"op_{i}" for i in range(len(df))]

metadatas = df[["Kalkış", "Hedef", "Durum", "Risk Skoru", "Araç Kontrolü", "Hava Şartları", "Trafik Yoğunluğu", "Mesafe (KM)", "Ortalama Günlük Sürüş Saati", "CO2 Emisyonu (KG)", "Araç Tipi"]].to_dict(orient="records")

embeddings = model.encode(documents, show_progress_bar=True)

collection.upsert(
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas,
    ids=op_ids
)

print(f"{len(documents)} tane simüle edilmiş operasyon verisi vektör veritabanına başarıyla eklendi.")