import streamlit as st
import pandas as pd
import plotly.express as px
import img2pdf
import requests
import os
import re
import glob
from agent import get_vector_collection, query_vector_database, run_logistic_agent
from sentence_transformers import SentenceTransformer
from report_history import load_report_history, save_report_history, clear_report_history, delete_report_by_index

BACKUP_FOLDER = "CSV_DATA"
if not os.path.exists(BACKUP_FOLDER):
    os.makedirs(BACKUP_FOLDER)

st.set_page_config(page_title="RotaLog AI Panel", page_icon="🚚", layout="wide")

if "processed_files" not in st.session_state:
    st.session_state["processed_files"] = set()

if "db_data" not in st.session_state:
    st.session_state["db_data"] = None

if "embedding_model" not in st.session_state:
    with st.spinner("Yapay Zeka Asistanı Hazırlanıyor... Lütfen Bekleyin ( İlk açılışta biraz sürebilir. )"):
        import torch
        
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        st.session_state["embedding_model"] = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
            device=device
        )
    st.rerun()

embedding_model = st.session_state["embedding_model"]

def markdown_to_pdf(markdown_text, route_name, params):
    from weasyprint import HTML
    
    html_content = markdown_text    
    html_content = re.sub(r'###\s*(.*?)(?=\n|$)', r'<h3>\1</h3>', html_content)    
    html_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_content)   
    html_content = re.sub(r'^-\s*(.*?)(?=\n|$)', r'• \1', html_content, flags=re.MULTILINE)
    html_content = html_content.replace('\n', '<br>')
    
    html_template = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{ size: A4; margin: 20mm; background-color: #ffffff; }}
            body {{ font-family: 'Arial', sans-serif; color: #1e293b; line-height: 1.6; font-size: 11pt; }}
            .header {{ background-color: #0f172a; color: #ffffff; padding: 22px; border-radius: 6px; margin-bottom: 25px; }}
            .header h1 {{ margin: 0; font-size: 18pt; letter-spacing: -0.5px; }}
            .header p {{ margin: 8px 0 0 0; color: #94a3b8; font-size: 10pt; }}
            h3 {{ color: #0284c7; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin-top: 28px; margin-bottom: 12px; font-size: 13pt; page-break-after: avoid; }}
            strong {{ color: #0f172a; font-weight: bold; }}
            br {{ margin-bottom: 6px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚚 RotaLog AI — Operasyon İyileştirme Raporu</h1>
            <p><strong>Sefer:</strong> {route_name} | <strong>Koşullar:</strong> {params}</p>
        </div>
        <div class="content">
            {html_content}
        </div>
    </body>
    </html>
    """
    return HTML(string=html_template).write_pdf()

@st.cache_resource
def load_vector_db():
    try:
        coll = get_vector_collection(collection_name="logistic_operations")
        return coll
    except Exception as e:
        st.error(f"Vektör veritabanına bağlanırken hata: {str(e)}")
        return None

operations_collection = load_vector_db()

# İlk veri kümesini çekme işlemini önbelleğe alıp CPU yükünü sıfırlıyoruz
@st.cache_resource
def fetch_initial_dataframe(_collection):
    if _collection is None:
        return pd.DataFrame()
    try:
        raw_data = _collection.get(include=['metadatas'])
        if raw_data and raw_data['metadatas']:
            return pd.DataFrame(raw_data['metadatas'])
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Veri tabanından veri çekilirken hata oluştu: {str(e)}")
        return pd.DataFrame()

if st.session_state["db_data"] is None:
    st.session_state["db_data"] = fetch_initial_dataframe(operations_collection)

df = st.session_state["db_data"]

st.title("🚚 RotaLog AI - Lojistik Yönetim ve Operasyon İyileştirme Paneli")
st.markdown("---")

st.subheader("📊 Filo Geçmiş Performans Analizi")

filtered_df = pd.DataFrame()

if not df.empty:
    with st.expander("🔍 Geçmiş Verileri Filtrele ve Grafik Analizini Göster", expanded=False):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            status_list = ["Hepsi"] + list(df["Durum"].unique()) if 'Durum' in df.columns else ["Hepsi"]
            selected_status = st.selectbox("Geçmiş Başarı Durumu", status_list)
            weather_list = ["Hepsi"] + list(df['Hava Şartları'].unique()) if 'Hava Şartları' in df.columns else ["Hepsi"]
            selected_weather = st.selectbox("Geçmiş Hava Şartları", weather_list)
        with f_col2:
            departure_list = ["Hepsi"] + list(df['Kalkış'].unique()) if 'Kalkış' in df.columns else ["Hepsi"]
            selected_departure = st.selectbox("Geçmiş Kalkış Noktası", departure_list)
            traffic_list = ["Hepsi"] + list(df['Trafik Yoğunluğu'].unique()) if 'Trafik Yoğunluğu' in df.columns else ["Hepsi"]
            selected_traffic = st.selectbox("Geçmiş Trafik Yoğunluğu", traffic_list)
        with f_col3:
            destination_list = ["Hepsi"] + list(df['Hedef'].unique()) if 'Hedef' in df.columns else ["Hepsi"]
            selected_destination = st.selectbox("Geçmiş Hedef Noktası", destination_list)
            vehicle_list = ["Hepsi"] + list(df['Araç Tipi'].unique()) if 'Araç Tipi' in df.columns else ["Hepsi"]
            selected_vehicle = st.selectbox("Geçmiş Araç Tipi", vehicle_list)
        with f_col4:
            check_list = ["Hepsi"] + list(df['Araç Kontrolü'].unique()) if 'Araç Kontrolü' in df.columns else ["Hepsi"]
            selected_check = st.selectbox("Geçmiş Araç Kontrol Durumu", check_list)

        temp_filtered_df = df.copy()
        if selected_status != "Hepsi" and 'Durum' in temp_filtered_df.columns:
            temp_filtered_df = temp_filtered_df[temp_filtered_df['Durum'] == selected_status]
        if selected_departure != "Hepsi" and 'Kalkış' in temp_filtered_df.columns:
            temp_filtered_df = temp_filtered_df[temp_filtered_df['Kalkış'] == selected_departure]
        if selected_destination != "Hepsi" and 'Hedef' in temp_filtered_df.columns:
            temp_filtered_df = temp_filtered_df[temp_filtered_df['Hedef'] == selected_destination]
        if selected_weather != "Hepsi" and 'Hava Şartları' in temp_filtered_df.columns:
            temp_filtered_df = temp_filtered_df[temp_filtered_df['Hava Şartları'] == selected_weather]
        if selected_traffic != "Hepsi" and 'Trafik Yoğunluğu' in temp_filtered_df.columns:
            temp_filtered_df = temp_filtered_df[temp_filtered_df['Trafik Yoğunluğu'] == selected_traffic]
        if selected_check != "Hepsi" and 'Araç Kontrolü' in temp_filtered_df.columns:
            temp_filtered_df = temp_filtered_df[temp_filtered_df['Araç Kontrolü'] == selected_check]
        if selected_vehicle != "Hepsi" and 'Araç Tipi' in temp_filtered_df.columns:
            temp_filtered_df = temp_filtered_df[temp_filtered_df['Araç Tipi'] == selected_vehicle]

        if not temp_filtered_df.empty:
            st.markdown("---")
            fig1, fig2, fig3, fig4, fig5 = None, None, None, None, None
            
            if 'Durum' in temp_filtered_df.columns:
                status_counts = temp_filtered_df['Durum'].value_counts().reset_index()
                status_counts.columns = ['Durum', 'Sefer Sayısı']
                fig1 = px.pie(status_counts, values="Sefer Sayısı", names="Durum", 
                              title="📈 Genel Sefer Başarı Dağılımı Ana Paneli",
                              color="Durum", color_discrete_map={"Başarılı": "#10b981", "Gecikmeli": "#f59e0b", "Başarısız/İptal": "#ef4444"},
                              hole=0.3)
                st.plotly_chart(fig1, use_container_width=True)
            
            st.markdown("---")
            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                if 'Araç Kontrolü' in temp_filtered_df.columns:
                    check_counts = temp_filtered_df['Araç Kontrolü'].value_counts().reset_index()
                    check_counts.columns = ['Kontrol', 'Sefer Sayısı']
                    fig2 = px.pie(check_counts, values="Sefer Sayısı", names="Kontrol", 
                                  title="🔧 Araç Kontrol Dağılımı",
                                  color="Kontrol", color_discrete_map={"Yapıldı": "#3b82f6", "Yapılmadı": "#64748b"})
                    st.plotly_chart(fig2, use_container_width=True)
            with row2_col2:
                if 'Trafik Yoğunluğu' in temp_filtered_df.columns:
                    traffic_counts = temp_filtered_df['Trafik Yoğunluğu'].value_counts().reset_index()
                    traffic_counts.columns = ['Trafik', 'Sefer Sayısı']
                    fig3 = px.pie(traffic_counts, values="Sefer Sayısı", names="Trafik", 
                                  title="🚦 Trafik Koşulları Dağılımı",
                                  color="Trafik", color_discrete_map={"Düşük": "#a7f3d0", "Orta": "#fef08a", "Yüksek": "#fed7aa", "Kilit": "#fca5a5"})
                    st.plotly_chart(fig3, use_container_width=True)
                    
            st.markdown("---")
            row3_col1, row3_col2 = st.columns(2)
            with row3_col1:
                if 'Hava Şartları' in temp_filtered_df.columns:
                    weather_counts = temp_filtered_df['Hava Şartları'].value_counts().reset_index()
                    weather_counts.columns = ['Hava Şartları', 'Sefer Sayısı']
                    fig4 = px.pie(weather_counts, values="Sefer Sayısı", names="Hava Şartları", 
                                  title="🌤️ Hava Şartları Dağılımı",
                                  color="Hava Şartları", color_discrete_map={"Güneşli": "#fde047", "Yağmurlu": "#60a5fa", "Karlı": "#e2e8f0", "Sisli": "#94a3b8", "Fırtınalı": "#334155"})
                    st.plotly_chart(fig4, use_container_width=True)
            with row3_col2:
                if 'Araç Tipi' in temp_filtered_df.columns:
                    vehicle_counts = temp_filtered_df['Araç Tipi'].value_counts().reset_index()
                    vehicle_counts.columns = ['Araç Tipi', 'Sefer Sayısı']
                    fig5 = px.pie(vehicle_counts, values="Sefer Sayısı", names="Araç Tipi", 
                                  title="🚛 Araç Tipi Dağılımı", color_discrete_sequence=px.colors.qualitative.Safe)
                    st.plotly_chart(fig5, use_container_width=True)
            
            try:
                images_bytes = []
                for f in [fig1, fig2, fig3, fig4, fig5]:
                    if f is not None:
                        images_bytes.append(f.to_image(format="png", width=800, height=500))
                if images_bytes:
                    st.markdown("---")
                    st.download_button(label="📥 Tüm Grafikleri Tek PDF Olarak İndir", data=img2pdf.convert(images_bytes), file_name="rotalog_performans_grafikleri.pdf", mime="application/pdf")
            except Exception:
                pass
        else:
            st.warning("⚠️ Seçtiğiniz filtre kombinasyonuna uygun geçmiş operasyon verisi bulunamadı. Lütfen filtreleri esnetin.")

    filtered_df = df.copy()
    if selected_status != "Hepsi" and 'Durum' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Durum'] == selected_status]
    if selected_departure != "Hepsi" and 'Kalkış' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Kalkış'] == selected_departure]
    if selected_destination != "Hepsi" and 'Hedef' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Hedef'] == selected_destination]
    if selected_weather != "Hepsi" and 'Hava Şartları' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Hava Şartları'] == selected_weather]
    if selected_traffic != "Hepsi" and 'Trafik Yoğunluğu' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Trafik Yoğunluğu'] == selected_traffic]
    if selected_check != "Hepsi" and 'Araç Kontrolü' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Araç Kontrolü'] == selected_check]
    if selected_vehicle != "Hepsi" and 'Araç Tipi' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Araç Tipi'] == selected_vehicle]

    st.markdown(f"**Veri Tabanından Çekilen Filtreli Sefer Sayısı:** `{len(filtered_df)}` / `{len(df)}`")

    m_col1, m_col2, m_col3 = st.columns(3)
    total_count = len(filtered_df)
    success_count = len(filtered_df[filtered_df['Durum'] == 'Başarılı']) if 'Durum' in filtered_df.columns else 0
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0
    delayed_count = len(filtered_df[filtered_df['Durum'] == 'Gecikmeli']) if 'Durum' in filtered_df.columns else 0
    
    with m_col1: st.metric(label="Filtreli Toplam Sefer", value=f"{total_count} Sefer")
    with m_col2: st.metric(label="Operasyon Başarı Oranı", value=f"%{success_rate:.1f}")
    with m_col3: st.metric(label="Gecikmeli Sefer Sayısı", value=f"{delayed_count} Sefer")
else:
    st.info("📥 Veri tabanında şu an aktif operasyonel veri bulunmuyor. Yapay zeka analizinin çalışabilmesi için lütfen aşağıdaki 'Grafik Analiz Paneli' sekmesinden ilk CSV dosyanızı import edin.")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Grafik Analiz Paneli", "➕ Yeni Sefer Planlama", "🗃️ Rapor Geçmişi"])

with tab1:
    st.markdown("#### 📊 Filo Geçmiş Performans Analizi")
    st.markdown("##### 📥 Yeni Sefer Verisi Yükle")
    uploaded_file = st.file_uploader("Şirket geçmiş sefer verilerini içeren CSV dosyasını seçin", type=["csv"], key="bulk_csv_uploader")
    
    if uploaded_file is not None:
        file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if file_identifier not in st.session_state["processed_files"]:
            try:
                new_data = pd.read_csv(uploaded_file)
                required_cols = ['Durum']
                
                if all(col in new_data.columns for col in required_cols):
                    backup_file_path = os.path.join(BACKUP_FOLDER, f"imported_{os.urandom(2).hex()}_{uploaded_file.name}")
                    new_data.to_csv(backup_file_path, index=False)
                    
                    with st.spinner("Yeni veriler yapay zeka hafızasına işleniyor..."):
                        new_documents = []
                        new_metadatas = []
                        new_ids = []
                        
                        for idx, row in new_data.iterrows():
                            departure_val = row.get("Kalkış", "Bilinmeyen")
                            destination_val = row.get("Hedef", "Bilinmeyen")
                            distance_val = row.get("Mesafe (KM)", 0)
                            weather_val = row.get("Hava Şartları", "Bilinmeyen")
                            traffic_val = row.get("Trafik Yoğunluğu", "Bilinmeyen")
                            check_val = row.get("Araç Kontrolü", "Bilinmeyen")
                            driving_time_val = row.get("Ortalama Günlük Sürüş Saati", 0)
                            automobile_type_val = row.get("Araç Tipi", "Bilinmeyen")
                            risk_score_val = row.get("Risk Skoru", 0.0)
                            status_val = row.get("Durum", "Bilinmeyen")
                            co2_emission_val = row.get("CO2 Emisyonu (KG)", 0.0)

                            processed_str = (
                                f"Kalkış noktası {departure_val}, hedefi {destination_val} olan ve {distance_val} KM mesafe kat eden operasyonda kullanılan araç tipi {automobile_type_val} olarak kaydedilmiştir. "
                                f"hava şartları {weather_val}, trafik yoğunluğu {traffic_val} olarak kaydedilmiştir. "
                                f"Yola çıkmadan önce araç kontrolü {check_val}. Sürücünün toplam sürüş saati {driving_time_val} saattir. "
                                f"Bu operasyonun risk skoru {float(risk_score_val):.2f} olup, süreç {status_val} durumla sonuçlanmıştır. "
                                f"Harcanan CO2 emisyonu {float(co2_emission_val):.2f} KG'dır."
                            )
                            new_documents.append(processed_str)
                            
                            new_metadatas.append({
                                "Kalkış": str(departure_val),
                                "Hedef": str(destination_val),
                                "Durum": str(status_val),
                                "Risk Skoru": float(risk_score_val),
                                "Araç Kontrolü": str(check_val),
                                "Hava Şartları": str(weather_val),
                                "Trafik Yoğunluğu": str(traffic_val),
                                "Mesafe (KM)": float(distance_val),
                                "Ortalama Günlük Sürüş Saati": float(driving_time_val),
                                "CO2 Emisyonu (KG)": float(co2_emission_val),
                                "Araç Tipi": str(automobile_type_val)
                            })
                            import os as python_os
                            new_ids.append(f"user_op_{python_os.urandom(3).hex()}_{idx}")
                        
                        new_embeddings = embedding_model.encode(new_documents).tolist()
                        operations_collection.add(
                            embeddings=new_embeddings,
                            documents=new_documents,
                            metadatas=new_metadatas,
                            ids=new_ids
                        )
                    
                    st.session_state["processed_files"].add(file_identifier)
                    st.session_state["db_data"] = None
                    fetch_initial_dataframe.clear() 
                    st.success(f"✅ {len(new_data)} adet yeni sefer verisi hafızaya ve grafiklere başarıyla eklendi!")
                    st.rerun()
                else:
                    st.error("❌ Yüklenen CSV dosyasında 'Durum' sütunu bulunamadı. Lütfen dosya formatını kontrol edin.")
            except Exception as e:
                st.error(f"Dosya işlenirken hata oluştu: {str(e)}")

with tab2:
    if df.empty:
        st.warning("⚠️ Yapay zeka modelinin geçmiş referans verilerine göre analiz yapabilmesi için öncelikle sisteme veri import edilmesi gerekmektedir. Lütfen yukarıdaki alandan ilk CSV dosyanızı import edin.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### 📋 Sefer Parametreleri")
            departure = st.selectbox("Kalkış Noktası", ["İzmir", "İstanbul", "Ankara"])
            destination = st.selectbox("Hedef Noktası", ["İstanbul", "İzmir", "Ankara"])
            weather = st.selectbox("Hava Durumu", ["Güneşli", "Yağmurlu", "Karlı", "Sisli", "Fırtınalı"])
            vehicle = st.selectbox("Araç Tipi", ["Tır", "Kamyonet", "Panel Van"])
            check = st.radio("Teknik Araç Kontrolü / Bakım", ["Yapıldı", "Yapılmadı"])
            
            coordinates = {"İzmir": [27.1287, 38.4192], "İstanbul": [28.9784, 41.0082], "Ankara": [32.8597, 39.9334]}
            start_coords = coordinates[departure]
            end_coords = coordinates[destination]
            
            if departure != destination:
                try:
                    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}?overview=full&geometries=geojson"
                    response = requests.get(osrm_url).json()
                    route_coords = response['routes'][0]['geometry']['coordinates']
                    route_df = pd.DataFrame(route_coords, columns=['lon', 'lat'])
                except Exception:
                    route_df = pd.DataFrame([{"lon": start_coords[0], "lat": start_coords[1]}, {"lon": end_coords[0], "lat": end_coords[1]}])
            else:
                route_df = pd.DataFrame([{"lon": start_coords[0], "lat": start_coords[1]}])

            markers_df = pd.DataFrame([{"City": f"Kalkış: {departure}", "lon": start_coords[0], "lat": start_coords[1]},
                                       {"City": f"Hedef: {destination}", "lon": end_coords[0], "lat": end_coords[1]}])

            fig_map = px.line_mapbox(route_df, lat="lat", lon="lon", zoom=5, height=350)
            fig_map.update_traces(line=dict(color="#ffa600", width=2))
            fig_points = px.scatter_mapbox(markers_df, lat="lat", lon="lon", hover_name="City", size_max=8, color_discrete_sequence=["#f43f5e"])
            fig_map.add_traces(fig_points.data)
            fig_map.update_layout(mapbox_style="carto-darkmatter", margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)
            st.plotly_chart(fig_map, use_container_width=True)
            
            st.markdown("---")
            analiz_butonu = st.button("🚀 Risk ve Maliyet Analizi Yap")

        with col2:
            st.markdown("#### 📊 Operasyon İyileştirme Raporu")
            if analiz_butonu:
                input_query = f"{weather} havada {departure}'den {destination}'e {vehicle} ile kargo seferi planlanıyor. Araç bakımı/kontrolü {check}."
                
                with st.spinner("Yapay Zeka Asistanınız raporunuzu hazırlanıyor..."):
                    try:
                        vector_results = query_vector_database(input_query, operations_collection, embedding_model)
                        rapor_output = run_logistic_agent(input_query, vector_results['distances'][0], vector_results['documents'][0])
                        st.markdown(rapor_output)
                        
                        current_history = load_report_history()
                        current_history.append({
                            "rota": f"{departure} -> {destination}",
                            "parametreler": f"{weather} / {vehicle}",
                            "icerik": rapor_output
                        })
                        save_report_history(current_history)
                    except Exception as e:
                        st.error(f"Analiz sırasında bir hata oluştu: {str(e)}")
                st.success("Rapor Başarıyla Üretildi!")
                with st.spinner("Raporunuz PDF formatına biçimlendiriliyor..."):
                    try:
                        pdf_data = markdown_to_pdf(rapor_output, f"{departure} -> {destination} ", f"{weather} Hava / {vehicle}")
                        st.download_button(label="📥 Bu Raporu PDF Olarak İndir", data=pdf_data, file_name=f"rotalog_rapor_{departure}_{destination}.pdf", mime="application/pdf")
                    except Exception as e:
                        st.error("Biçimlendirme sırasında bir hata oluştu...")    
            else:
                st.info("Sol taraftaki menülerden yeni sefer şartlarını seçip 'Analiz Yap' butonuna basın.")

with tab3:
    st.markdown("#### 🗃️ Üretilen Rapor Geçmişi")
    saved_reports = load_report_history()
    if saved_reports:
        if "show_confirm" not in st.session_state:
            st.session_state.show_confirm = False
        if not st.session_state.show_confirm:
            if st.button("🗑️ Rapor Geçmişini Tamamen Temizle", type="primary"):
                st.session_state.show_confirm = True
                st.rerun()
        else:
            st.error("⚠️ Tüm geçmişi silmek istediğinize emin misiniz? Bu işlem geri alınamaz!")
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                if st.button("✅ Evet, Eminim", type="primary", use_container_width=True):
                    clear_report_history()
                    st.session_state.show_confirm = False
                    st.success("Tüm rapor geçmişi lokal cihazdan kalıcı olarak silindi!")
                    st.rerun()
            with c_col2:
                if st.button("❌ Hayır, İptal Et", use_container_width=True):
                    st.session_state.show_confirm = False
                    st.rerun()

        st.markdown("---")
        for idx in reversed(range(len(saved_reports))):
            item = saved_reports[idx]
            r_col1, r_col2 = st.columns([9, 1])
            with r_col1:
                with st.expander(f"📌 Sefer: {item['rota']} | Şartlar: {item['parametreler']}"):
                    st.markdown(item['icerik'])
                    st.markdown("---")
                    try:
                        hist_pdf_data = markdown_to_pdf(item['icerik'], item['rota'], item['parametreler'])
                        st.download_button(label="📥 Bu Raporu PDF Olarak İndir", data=hist_pdf_data, file_name=f"rotalog_gecmis_{idx}.pdf", mime="application/pdf", key=f"pdf_hist_{idx}")
                    except Exception as e:
                        st.warning("PDF motoru hatası nedeniyle bu rapor için ham metin indirme aktif edildi.")
                        st.download_button(label="📥 Bu Raporu TXT Olarak İndir", data=item['icerik'].encode('utf-8'), file_name=f"rotalog_gecmis_{idx}.txt", mime="text/plain", key=f"txt_hist_{idx}")
            with r_col2:
                if st.button("❌", key=f"del_{idx}"):
                    delete_report_by_index(idx)
                    st.rerun()
    else:
        st.info("Sistemde henüz üretilmiş veya kaydedilmiş bir rapor geçmişi bulunmuyor.")