# MVTec AD transistör denetimi: YOLO + PatchCore

[MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) veri setinin `transistor` kategorisi üzerinde iki aşamalı bir görsel kalite kontrol deneyi.

| Aşama | Model | Soru | NG nedeni |
|---|---|---|---|
| 1 | YOLO11n + geometri kuralları | Parça var mı, doğru yerde ve doğru yönde mi? | parça yok · bacak yok · yön hatalı (>15°) · kaymış |
| 2 | PatchCore (Anomalib) | Yerindeki parçada kusur var mı? | skor ≥ 0.5 |

Eğitimde **sadece sağlam parçalar** kullanıldı. Kusurlu görüntüler ve maskeler yalnızca testte kullanılıyor. YOLO etiketleri de sağlam görüntülerden otomatik üretildi, elle etiketleme yapılmadı.

## Sonuçlar

Test kümesi: 60 sağlam ve 40 kusurlu görüntü (bükük bacak, kesik bacak, hasarlı gövde, yanlış yerleşim; her birinden 10).

| | Sadece PatchCore | YOLO + PatchCore |
|---|---|---|
| Kaçırılan kusurlu | 2/40 | **0/40** |
| Yanlış alarm | 1/60 | 1/60 |
| Doğruluk | %97.0 | **%99.0** |

PatchCore tek başına image AUROC 0.995, pixel AUROC 0.973 verdi. YOLO doğrulama kümesinde mAP50 0.995 verdi.

**Gözlemler**
- PatchCore yerel yamalara baktığı için transistörün **hiç olmadığı** boş kartları kaçırıyor. Boş bakırın her yaması eğitimde zaten görülmüş oluyor. Bu iki örneği YOLO yakaladı.
- Ters ya da yan takılmayı en iyi ayıran kural bacak açısı. Sağlam parçalarda açı 80.5°–98.7° aralığında kalıyor.
- Konum kuralı zayıf. Sağlam parçalar kart üzerinde geniş bir alana dağılmış, `misplaced/000` örneği toleransı kıl payı geçti (0.115'e karşı 0.111).
- Kalan tek yanlış alarm (`good/014`) kartın üstündeki bir kıl ya da çizik. Etikette "sağlam" yazıyor ama model gerçek bir tuhaflık yakalıyor.
- Sayılar küçük: yanlış yerleşim türünden sadece 10 örnek var. Sonuçlar bu veri setine özgü.

## Kurulum

Python 3.11 ve CUDA'lı bir GPU gerekiyor. Deney RTX 4060 Laptop (8 GB) üzerinde WSL2 altında yapıldı.

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

Anomalib'in otomatik indirmesi şu an çalışmıyor ([anomalib#3892](https://github.com/open-edge-platform/anomalib/issues/3892)). Veri setini MVTec'in sitesinden indirip açın:

```bash
mkdir -p datasets/MVTecAD && tar -xf mvtec_anomaly_detection.tar.xz -C datasets/MVTecAD
```

## Çalıştırma

```bash
python scripts/train_patchcore.py --category transistor     # ~2 dk
python scripts/yolo_autolabel.py                            # train/good → body/leads etiketleri
python scripts/train_yolo.py                                # ~7 dk
python scripts/inspect_two_stage.py --calibrate             # geometri eşikleri, sadece train/good'dan
python scripts/inspect_two_stage.py --input datasets/MVTecAD/transistor/test
```

Her görüntü için panel (YOLO kutuları | ısı haritası | karar) ve her görüntünün hangi aşamada ve neden kaldığını gösteren `report.csv`, `results/inspection_2stage/transistor/` altına yazılır.

| Betik | İş |
|---|---|
| `train_patchcore.py` | Bir MVTec kategorisinde PatchCore eğitir ve test eder |
| `inspect_parts.py` | Sadece PatchCore ile OK/NG kararı verir, kusur bölgesini kutular |
| `yolo_autolabel.py` | Sağlam görüntülerden gövde/bacak etiketlerini otomatik üretir |
| `train_yolo.py` | YOLO11n'i geniş augmentasyonla eğitir (döndürme, çevirme, kaydırma) |
| `inspect_two_stage.py` | İki aşamalı denetim, kalibrasyon ve rapor |
| `yolo_smoke.py` | Kurulumu hızlıca test eder |

## Lisans notu

MVTec AD [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) lisanslı, ticari kullanımı yasak. Veri seti, model ağırlıkları ve çıktılar `.gitignore` ile repoya dahil edilmiyor.
