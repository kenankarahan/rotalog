import random
import datetime
import pandas as pd


# Defining our parameter pool for the simulation

distance_dict = {
    ("İstanbul", "İzmir"): 480,
    ("İzmir", "İstanbul"): 480,
    ("Ankara", "İzmir"): 590,
    ("İzmir", "Ankara"): 590,
    ("İstanbul", "Ankara"): 450,
    ("Ankara", "İstanbul"): 450
}

risk_scores = {
    "Hava Şartları": {
        "Güneşli": 0,
        "Yağmurlu": 10,
        "Karlı": 30,
        "Sisli": 20,
        "Fırtınalı": 30
    },
    "Trafik Yoğunluğu": {
        "Düşük": 0,
        "Orta": 10,
        "Yüksek": 20
    },
    "Araç Kontrolü" : {
        "Yapıldı" : 0,
        "Yapılmadı" : 20
    }
}

automobile_types_with_co2 = {
    "Tır" : 1,
    "Panel Van" : 0.2,
    "Kamyonet" : 0.6,
}



# Functions to calculate our parameters

def co2_emission(distance, automobile_type):
    co2_per_km = automobile_types_with_co2[automobile_type]
    return distance * co2_per_km

def average_driving_time(distance, average_speed=100):
    
    ddt = random.randint(24, 48)
    
    driving_time = distance / average_speed
    
    risk_score = 0
    
    if driving_time > 8:
        rdt = (driving_time - 8) + 24 
    else:
        rdt = driving_time
    
    if rdt > ddt:
        risk_score = 30
    
    result = driving_time + risk_score
    
    return result, driving_time

def simulate(risk_scores, departure, destination, distance, automobile_type):
    
    # Calculating our risk scores first.
    
    risk_score = 0
    risk_factor = random.uniform(0.1, 1.0)
    weather = random.choice(list(risk_scores["Hava Şartları"].keys()))
    weather_score = risk_scores["Hava Şartları"][weather]
    traffic = random.choice(list(risk_scores["Trafik Yoğunluğu"].keys()))
    traffic_score = risk_scores["Trafik Yoğunluğu"][traffic]
    check = random.choice(list(risk_scores["Araç Kontrolü"].keys()))
    check_score = risk_scores["Araç Kontrolü"][check]
    driving_time = average_driving_time(distance)
    risk_score = weather_score + traffic_score + check_score + (driving_time[0] * risk_factor)
    
    if risk_score <= 35:
        status = "Başarılı"
    elif 35 < risk_score <= 65:
        status = "Gecikmeli"
    else:
        status = "Başarısız / İptal"
     
      
    # Calculating for CO2 emissions next.
    
    co2 = co2_emission(distance, automobile_type)
    
    return {
    "Kalkış": departure,
    "Hedef": destination,
    "Araç Tipi": automobile_type,
    "Mesafe (KM)": distance,
    "Hava Şartları": weather,
    "Trafik Yoğunluğu": traffic,
    "Araç Kontrolü": check,
    "Ortalama Günlük Sürüş Saati": driving_time[1],
    "Risk Faktörü": risk_factor,
    "Risk Skoru": risk_score,
    "Durum": status,
    "CO2 Emisyonu (KG)": co2
    }
    
    
data_pool = []
    
for i in range(5000):
    departure, destination = random.choice(list(distance_dict.keys()))
    distance = distance_dict[(departure, destination)]
    automobile_type = random.choice(list(automobile_types_with_co2.keys()))
    
    parameters = simulate(risk_scores, departure, destination, distance, automobile_type)
    
    data_pool.append(parameters)
df = pd.DataFrame(data_pool)
df.to_csv("simulated_data.csv", index=False, encoding="utf-8-sig")