# Deep Image Prior for Inverse Problems in Imaging

Implementazione completa in **Python + PyTorch** containerizzata con **Docker** del metodo **Deep Image Prior (DIP)** (Ulyanov et al., CVPR 2018) per risolvere problemi inversi di imaging **senza alcun dataset di training né pre-training**.

---

## Formulazione Matematica

Dato un operatore diretto noto $A$ e una misura corrotta $y = A(x) + \epsilon$, il Deep Image Prior parametrizza l'immagine incognita come output di una rete neurale convoluzionale $x = f_\theta(z)$ con input fisso $z$ (rumore o griglia di coordinate) e ottimizza direttamente i pesi $\theta$:

$$\theta^* = \arg\min_\theta \; \mathcal{L}\big(A(f_\theta(z)),\; y\big) + \lambda_{\mathrm{TV}}\,\mathrm{TV}(f_\theta(z))$$

La struttura stessa della rete agisce come *prior* implicito verso immagini naturali.

---

## Problemi Inversi Implementati

| Problema Inverso | Operatore Diretto $A(x)$ | Note |
|---|---|---|
| **Denoising** | $A(x) = x$ | Rimozione rumore Gaussiano con EMA & input noise regularization |
| **Inpainting** | $A(x) = M \odot x$ | Maschere `random`, `block` (buco centrale) e `stripes` |
| **Deblurring (+ Denoising)** | $A(x) = k * x$ | Kernel `gaussian` e `motion` blur + rumore additivo opzionale |
| **Super-Resolution** | $A(x) = \mathrm{Downsample}_s(x)$ | Downsampling anti-aliased differenziabile ($2\times, 4\times$) |
| **X-Ray Tomography (CT)** | $A(x) = \mathcal{R}(x)$ | Trasformata di Radon differenziabile a fascio parallelo (sinogramma) |
| **Compressed Sensing** | $A(x) = \Phi x$ | Matrice Gaussiana casuale a blocchi (`random_matrix`) o FFT sottocampionata (`fourier`) |

### Architetture & Input Supportati
- **Architetture**: `SkipNet` (hourglass encoder-decoder con skip connections dal paper DIP) e `UNet`.
- **Input fisso $z$**: `uniform`, `normal`, `meshgrid` (coordinate 2D).
- **Tecniche di stabilizzazione**: Exponential Moving Average (EMA) dell'output, perturbazione dell'input (`reg_noise_std`), e regolarizzazione opzionale Total Variation (`tv_weight`).

---

## Avvio Rapido con Docker

### 1. Interfaccia Web Interattiva (Streamlit)

```bash
docker compose up --build
```

Apri il browser su **[http://localhost:8501](http://localhost:8501)**.

Per avviare in background:
```bash
docker compose up -d --build
```

Per fermare il container:
```bash
docker compose down
```

### 2. Esecuzione da CLI (Esperimenti Riproducibili)

I grafici e i file JSON delle metriche vengono salvati automaticamente nella cartella locale `./outputs/`:

```bash
# Denoising
docker compose --profile cli run --rm dip-cli \
  --problem denoising --image astronaut --iterations 600 --noise-std 0.12

# X-Ray Tomography (Shepp-Logan phantom)
docker compose --profile cli run --rm dip-cli \
  --problem tomography --image shepp_logan --num-angles 30 --iterations 600 --tv-weight 0.001

# Compressed Sensing (30% misure)
docker compose --profile cli run --rm dip-cli \
  --problem compressed_sensing --image camera --cs-ratio 0.30 --iterations 600
```

### 3. Eseguire i Test Automatici nel Container

```bash
docker compose run --rm --entrypoint pytest dip-app -v
```

### 4. Supporto GPU NVIDIA (Opzionale)

Di default il `Dockerfile` installa la wheel CPU di PyTorch (leggera e compatibile ovunque). Se hai una GPU NVIDIA con `nvidia-container-toolkit`:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 docker compose up --build
```

---

## Struttura del Progetto

```text
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── src/dip/
│   ├── models.py       # SkipNet (paper DIP) & UNet
│   ├── inputs.py       # Generatori input z (uniform, normal, meshgrid)
│   ├── operators.py    # Operatori diretti differenziabili A(x)
│   ├── metrics.py      # PSNR, SSIM, Total Variation
│   ├── data.py         # Immagini campione offline & I/O
│   └── solver.py       # Motore di ottimizzazione DIP con EMA & callback
├── app/
│   └── streamlit_app.py  # Web UI interattiva
├── scripts/
│   └── run_experiment.py # Runner CLI riproducibile
└── tests/
    └── test_dip.py     # Suite pytest (operatori, gradienti, end-to-end)
```
