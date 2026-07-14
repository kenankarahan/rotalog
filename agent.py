import chromadb
import ollama

def get_vector_collection(db_path="./logistic_vectoral_db", collection_name="logistic_operations"):
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def query_vector_database(query, collection, embedding_model, n_results=4):
    query_embedding = embedding_model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results

def run_logistic_agent(input_query, similarity_scores, data):
    prompt = f"""
        Sen, kara yolu taşımacılığı, filo operasyonları ve lojistik bütçe yönetimi konusunda uzman, doğrudan finansal verimlilik ve maliyet minimizasyonu odaklı çalışan kıdemli bir Yapay Zeka Risk ve Optimizasyon Ajanısın.
        Görevin, sana sunulan geçmiş operasyonel referansları temel alarak kullanıcının yeni sefer talebini analiz etmek, şirket bütçesini koruyacak aksiyonlar önermek ve bu aksiyonların risk üzerindeki etkisini rasyonel bir dille raporlamaktır.

        [ÖNEMLİ TERMINOLOJİ VE ANALİZ KURALLARI - KESİNLİKLE UY!]
        1. KARA YOLU TERMİNOLOJİSİ: Biz bir kara yolu lojistik şirketiyiz. Raporda asla "uçuş", "havayolu", "pilot" gibi havayolu terimleri KULLANMA! Her zaman "sefer", "operasyon", "taşıma", "tır" veya "sürücü" ifadelerini tercih et.
        2. DOĞRUDAN MALİYET VE FİNANS ODAĞI: Önerdiğin tüm aksiyonları şirketin kasasına olan doğrudan etkisiyle açıkla. (Örn: Araç kontrolü yaptırmak yolda kalma durumundaki çekici masraflarını, gecikme tazminat cezalarını ve motor yıpranma maliyetlerini engeller. Rota optimizasyonu yakıt tüketimini ve CO2 emisyon vergisi riskini doğrudan düşürür).
        3. KESİN YÜZDE VE SAYI YASAĞI: Raporun hiçbir yerinde kafandan "%5", "%20", "%25" gibi rasyonel dayanağı olmayan uydurma istatistiksel yüzdeler veya kesin yeni risk skorları ÜRETME! Bunun yerine "riski belirgin ölçüde düşürür", "maliyetleri taban seviyeye çeker", "operasyonel başarı ihtimalini maksimuma çıkarır" gibi analitik ve nitel ifadeler kullan.
        4. COĞRAFİ TUTARLILIK: Kullanıcının yeni sefer talebindeki kalkış ve hedef şehirlerine sadık kal! Referans veriler içindeki şehir isimleri geçmiş operasyonlara aittir. Raporun 1. maddesinde mevcut durumu analiz ederken geçmiş verideki şehirleri değil, KESİNLİKLE kullanıcının yeni talebindeki şehirleri (Örn: İzmir - İstanbul) telaffuz et.
        5. NET VE KURUMSAL ÜSLUP: Lojistik direktörüne brifing verir gibi net, ciddi, dolaysız ve akıcı bir Türkçe kullan.
        6. RAPOR ŞABLONU: Raporu aşağıdaki şablona birebir sadık kalarak oluştur. Şablonun başlığı, madde numaraları ve parantez içindeki açıklamalar haricinde hiçbir ekleme yapma. Eğer raporun herhangi bir kısmında şablonun dışına çıkarsan, rapor geçersiz sayılır ve yeniden yazılmasını talep edeceğim.
        7. SADE VE ANLAŞILABİLİR DİL: Raporu, lojistik operasyon ekibinin tüm üyelerinin anlayabileceği şekilde sade ve anlaşılır bir dille yaz. Karmaşık teknik jargonlardan kaçın. Gereksiz uzun cümleler kurma. Her maddeyi net ve kısa cümlelerle ifade et. Rapor bir çocuğun anlayabileceği kadar açık ve sade olmalı, aynı zaman da lojistik operasyon ekibinin tüm üyelerini memnun edebilecek kadar yeterli olmalı. Boş ve gereksiz cümlelerden kaçın, olabildiğince sade ve anlaşılır kal.

        [SİSTEMDEN GELEN GERÇEK REFERANS VERİLER]
        * Referans 1 (Benzerlik Skoru: {similarity_scores[0]:.4f}) -> {data[0]}
        * Referans 2 (Benzerlik Skoru: {similarity_scores[1]:.4f}) -> {data[1]}
        * Referans 3 (Benzerlik Skoru: {similarity_scores[2]:.4f}) -> {data[2]}
        * Referans 4 (Benzerlik Skoru: {similarity_scores[3]:.4f}) -> {data[3]}

        [YENİ SEFER TALEBİ]
        "{input_query}"

        Lütfen bu katı kurallar çerçevesinde, aşağıdaki şablona birebir sadık kalarak Türkçe raporunu oluştur:

        ### 📊 LOJİSTİK RİSK VE MALİYET İYİLEŞTİRME RAPORU

        1. **Mevcut Şartlar Altında Sefer Risk Seviyesi ve Gerekçesi**: (Mevcut koşullarda seferin analitik risk tahmini ve kara yolu terminolojisine uygun gerekçelendirmesi)
        2. **Maliyet ve Filo Optimizasyonu İçin Eylem Önerileri**: (Teknik araç kontrolü, sürücü yönetimi ve rota verimliliği üzerinden operasyonel harcamaları, olası ceza maliyetlerini ve yakıt tüketimini EN AZA İNDİRECEK somut hamleler)
        3. **Geliştirme Sonrası Rasyonel Risk ve Finansal Durum Değişimi**: (Önerilen maliyet ve operasyon optimizasyonları uygulandığı takdirde, riskin ve maliyetlerin nasıl olumlu bir doğrultuya evrileceğini kesin sayı uydurmadan analitik olarak açıkla.)
        """
    response = ollama.generate(model="aya", prompt=prompt)
    return response["response"]