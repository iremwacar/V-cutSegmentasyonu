# Vücut Segmentasyonu Web

Bir fotoğraf üzerindeki kişiyi odaklayıp arka planı bulanıklaştırır, MediaPipe Pose noktalarına dayalı bölgesel ölçekleme ile vücut şekillendirme (zayıf/kilolu) uygular. Bu repo, FastAPI tabanlı basit bir web arayüzü (yükle ve +/-) içerir.

## Özellikler
- Görsel yükle (JPG/PNG)
- +/- butonları ile zayıf/kilolu seviyesi (-10 ... 0 ... +10)
- YOLOv8 ile ana kişiyi bulma, MediaPipe Selfie Segmentation ile arka planı bulanıklaştırma
- MediaPipe Pose ile bölgesel deformasyon (göğüs, bel, kalça, kollar, bacaklar vb.)
- Bölge tepkileri ve guardrail ile daha doğal sonuçlar (zayıflarken göğüsün büyüme artefaktını önleme)
- Modüler frontend (public/) ve API (FastAPI)

## Kurulum
1. Python 3.10+ önerilir.
2. Bağımlılıkları yükleyin:
```bash
python -m pip install -r requirements.txt
```

## Çalıştırma
```bash
python server.py
```
Tarayıcıdan `http://localhost:8001` adresine gidin.

## Kullanım
1. Görsel yükleyin.
2. Ağırlık seviyesini `-10` ile `+10` arasında ayarlamak için `- / +` butonlarını kullanın.
3. `Sıfırla` ile tekrar 0 seviyesine dönün.

## API
- POST `/api/upload` (multipart/form-data: file)
  - Döner: `{ "image_id": "..." }`
- GET `/api/process?image_id=...&level=INT`
  - Döner: `{ "base64": "..." }` (JPEG Base64)

## Mimari
- `pipeline.py`: ModelRegistry, odak+mask işlemi, bölge ölçek hesaplama, deformasyon hattı.
- `deform_body.py`: Pose landmark çıkarımı ve RBF tabanlı deformasyon.
- `focus_person.py`: Orijinal CLI akışındaki odak/segmentasyon (referans).
- `server.py`: FastAPI sunucusu, statik dosya servis ve API uçları.
- `public/`: `index.html`, `app.css`, `app.js` modüler arayüz.

## Özelleştirme
- Bölge tepkileri: `SLIM_RESPONSE_WEIGHT` ve `FAT_RESPONSE_WEIGHT` sözlükleri ile bölgesel etkileri ayarlayın.
- Guardrail: Zayıflama durumunda göğüs ölçeği, `shoulders` ve `waist` ortalamasını geçmeyecek şekilde sınırlandırılır. İncelik için `pipeline.py` içinde düzenleyin.
- Maksimum seviye: `compute_scale_dict_from_level` fonksiyonundaki `max_level` ile kontrol edilir.

## Notlar
- İlk isteklerde model yüklemeleri nedeniyle gecikme olabilir. Modeller `ModelRegistry` ile bir kez yüklenir.
- Sonuçlar, görüntü çözünürlüğüne ve poz algılamanın doğruluğuna bağlıdır.
- Bu demo, bellek içi `ImageStore` kullanır. Üretimde kalıcı depolama ve kuyruklama önerilir. 