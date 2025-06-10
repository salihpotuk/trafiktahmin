import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os
import joblib

# 1. Dosya yolunu kontrol et
csv_file = 'Proje_Sokagi_Trafik_Hava_Verisi_Ocak2025.csv'
if not os.path.exists(csv_file):
    print(f"Hata: '{csv_file}' dosyası bulunamadı. Lütfen dosyayı '{os.getcwd()}' dizinine taşıyın veya tam yolunu belirtin.")
    exit()

# 2. Veri setini oku
df = pd.read_csv(csv_file)

# 3. Veri ön işleme
# Saat sütununu sayısal değere çevir (örneğin, "08:00" -> 8)
def extract_hour(time_str):
    try:
        return int(time_str.split(':')[0])
    except (AttributeError, ValueError):
        return None

df['Saat'] = df['Saat'].apply(extract_hour)

# Hava Durumu sütununu ayır: Hava durumu ve sıcaklık
def split_weather(weather_str):
    try:
        condition, temp = weather_str.split(', ')
        temp = int(temp.replace('°C', ''))
        return condition, temp
    except (AttributeError, ValueError):
        return None, None

df[['Hava Durumu', 'Sıcaklık']] = pd.DataFrame(df['Hava Durumu'].apply(split_weather).tolist(), index=df.index)

# Kategorik verileri kodla
le_day = LabelEncoder()
le_weather = LabelEncoder()
le_traffic = LabelEncoder()

df['Gün'] = le_day.fit_transform(df['Gün'])
df['Hava Durumu'] = le_weather.fit_transform(df['Hava Durumu'].astype(str))
df['Trafik Durumu'] = le_traffic.fit_transform(df['Trafik Durumu'])

# 4. Eksik verileri kontrol et ve doldur
df = df.fillna(df.mode().iloc[0])

# 5. Özellikler (X) ve hedef (y) ayır
# Tarih sütununu kullanmıyoruz
X = df[['Gün', 'Saat', 'Hava Durumu', 'Sıcaklık']]
y = df['Trafik Durumu']

# 6. Veriyi eğitim ve test setlerine böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Modeli oluştur ve eğit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Modeli test et
y_pred = model.predict(X_test)
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred, target_names=le_traffic.classes_))

# 9. Yeni bir tahmin yap
# Örnek: Pazartesi, saat 08:00, yağmurlu, 10°C
yeni_veri = pd.DataFrame({
    'Gün': [le_day.transform(['Saturday'])[0]],
    'Saat': [8],
    'Hava Durumu': [le_weather.transform(['Yağmurlu'])[0]],
    'Sıcaklık': [2]
})
tahmin = model.predict(yeni_veri)
tahmin_etiket = le_traffic.inverse_transform(tahmin)
print(f"Tahmini trafik durumu: {tahmin_etiket[0]}")

os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/traffic_model.pkl')
joblib.dump(le_day, 'model/le_day.pkl')
joblib.dump(le_weather, 'model/le_weather.pkl')
joblib.dump(le_traffic, 'model/le_traffic.pkl')
print("Model ve encoder'lar 'model/' dizinine kaydedildi.")