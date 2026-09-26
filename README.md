[README2.md](https://github.com/user-attachments/files/32680773/README2.md)
# IMDB sharhlari bo'yicha sentiment tahlili — DistilBERT fine-tuning

Film sharhining ijobiy yoki salbiy ekanini aniqlaydigan model. Oldindan o'qitilgan
`distilbert-base-uncased` Transformer modeli IMDB datasetida Hugging Face `Trainer`
yordamida qo'shimcha o'qitildi (fine-tuning).

## Loyiha haqida qisqacha

| | |
|---|---|
| Vazifa | Binar matn klassifikatsiyasi (0 — negative, 1 — positive) |
| Model | `distilbert-base-uncased` + klassifikatsiya boshi (`AutoModelForSequenceClassification`, `num_labels=2`) |
| Dataset | [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) |
| Kutubxonalar | `transformers`, `datasets`, `torch` |
| Muhit | Google Colab, GPU (Tesla T4) |
| O'qitish vaqti | ~5.5 daqiqa (564 qadam) |

## Dataset

IMDB datasetida 25 000 ta train va 25 000 ta test sharhi bor, sinflar teng taqsimlangan.
Tez tajriba o'tkazish uchun ma'lumotlardan tasodifiy kichik qism olindi:

- **Train:** 3 000 ta sharh (`shuffle(seed=42)`)
- **Test (validatsiya):** 300 ta sharh (`shuffle(seed=42)`)

> Eslatma: `datasets` kutubxonasining yangi versiyalarida qisqa nom `imdb` ishlamaydi,
> to'liq nom `stanfordnlp/imdb` ko'rsatilishi kerak.

## Ish jarayoni

1. **Kutubxonalarni o'rnatish.** `datasets`, `fsspec` va `transformers` eng so'nggi versiyalarga yangilandi.
2. **Datasetni yuklash.** `load_dataset("stanfordnlp/imdb")`, keyin train va test'dan kichik qismlar ajratildi.
3. **Tokenizatsiya.** `AutoTokenizer` (`distilbert-base-uncased`), `truncation=True`,
   `padding="max_length"`, `max_length=256`. `.map(batched=True)` bilan butun datasetga qo'llandi.
4. **Model.** `AutoModelForSequenceClassification.from_pretrained(..., num_labels=2)`.
   DistilBERT tanasi tayyor holda yuklanadi. Klassifikatsiya qatlamlari (`pre_classifier`, `classifier`)
   esa yangidan yaratiladi va aynan shu vazifa uchun o'qitiladi.
5. **O'qitish.** Hugging Face `Trainer`:

   ```python
   TrainingArguments(
       output_dir="./results",
       num_train_epochs=3,
       per_device_train_batch_size=16,
       per_device_eval_batch_size=64,
       logging_steps=10,
       eval_strategy="steps",   # har 10 qadamda validatsiya
       report_to="none",
   )
   ```

6. **Inference.** Yangi matnlar tokenizatsiya qilinadi, model logitlari `softmax` orqali
   ehtimolga aylantiriladi va `argmax` bilan sinf tanlanadi.

## Natijalar

### Loss dinamikasi

| Qadam | Epoch | Training loss | Validation loss |
|---|---|---|---|
| 10 | ~0.05 | 0.687 | 0.681 |
| 100 | ~0.5 | 0.282 | 0.344 |
| **180** | **~1.0** | **0.356** | **0.317 (eng past)** |
| 300 | ~1.6 | 0.159 | 0.368 |
| 440 | ~2.3 | 0.039 | 0.476 |
| 564 | 3.0 | 0.149 | 0.451 |

- O'rtacha training loss: **0.237**
- Eng yaxshi validation loss: **0.317** (≈180-qadam, 1-epoch oxiri)
- Yakuniy validation loss: **0.451**

**Xulosa:** 1-epochdan keyin model **overfitting**ga o'tadi. Training loss 0.04–0.1 gacha tushadi,
validation loss esa 0.32 dan 0.45–0.6 gacha ko'tariladi. 3 000 ta namunali kichik datasetda
1 epoch yetarli bo'lgan.

### Yangi matnlarda sinov

| Matn | Bashorat | Ishonch |
|---|---|---|
| I love this movie | positive | 0.9946 |
| This movie was absolutely awful | negative | 0.9966 |
| Just ok movie, not so bad | negative | 0.9949 |
| Movie just bad, or has advantages | negative | 0.9972 |

Aniq ijobiy va aniq salbiy sharhlar to'g'ri aniqlandi. Aralash yoki neytral sharh
("Just ok movie, not so bad") esa juda yuqori ishonch bilan salbiy deb baholandi.
Buning sababi — model faqat ikki sinfni biladi va overfitting tufayli haddan tashqari
ishonchli bo'lib qolgan.

## Cheklovlar

- Datasetning faqat ~12% qismi (3 000 / 25 000) ishlatildi. Test to'plami ham kichik (300 ta).
- Faqat loss kuzatildi: accuracy, precision, recall va F1 hisoblanmadi.
- Oxirgi (overfit bo'lgan) checkpoint ishlatildi, eng yaxshisi emas.
- Neytral sinf yo'q: aralash sharhlar majburan ikki sinfdan biriga tushadi.

## Keyingi qadamlar

- [ ] `compute_metrics` qo'shib, accuracy va F1 ni hisoblash
- [ ] `load_best_model_at_end=True` va `metric_for_best_model="eval_loss"` bilan eng yaxshi checkpointni saqlash
- [ ] `EarlyStoppingCallback` yoki `num_train_epochs=1`–`2`
- [ ] To'liq 25 000 ta train sharhida o'qitish
- [ ] `padding="max_length"` o'rniga dinamik padding (`DataCollatorWithPadding` allaqachon bor) — tezroq o'qitish uchun
- [ ] Modelni `trainer.save_model()` bilan saqlab, Hugging Face Hub'ga yuklash

## Ishga tushirish

1. Notebookni Google Colab'da oching.
2. **Runtime → Change runtime type → GPU (T4)** ni tanlang.
3. Katakchalarni ketma-ket ishga tushiring.

```bash
pip install -U datasets fsspec transformers
```
