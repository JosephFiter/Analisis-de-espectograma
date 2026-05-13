# Ayudantia Itba — Instrucciones

## ¿Qué hace el programa?

1. **Cargás** un video (`.avi`, `.mp4`, etc.) y un audio (`.wav`) de la rata.  
2. **Configurás** el espectrograma: colormap, rango de dB, rango de frecuencias, FFT, contraste, etc.  
3. **Apretás** "Generar Video" y el programa produce un **nuevo video** que tiene el espectrograma incrustado (abajo-izquierda por defecto) con una línea roja que avanza sincronizada con el tiempo.

---

## Requisitos previos

- Python **3.9, 3.10, 3.11 o 3.12** (64-bit).  
  Descargá desde https://python.org/downloads/ y marcá **"Add Python to PATH"** durante la instalación.
- Windows 10 / 11 64-bit.

---

## Instalación (una sola vez)

Abrí **PowerShell** y ejecutá:

```powershell

python -m venv .venv
.venv\Scripts\Activate.ps1
```

> Si da error de permisos:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Luego volvé a activar el entorno.

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

La primera instalación tarda 2–5 minutos (librosa + numpy + opencv).

### Verificación
```powershell
python -c "import PyQt5, cv2, librosa; print('Todo OK')"
```
Debe mostrar: `Todo OK`

---

## Cómo correr el programa

```powershell
.venv\Scripts\Activate.ps1
python main.py
```

---

## Uso paso a paso

### Paso 1 — Cargar archivos
- **Cargar Video…** → seleccioná tu `.avi` (o `.mp4`, `.mkv`, etc.)
- **Cargar Audio…** → seleccioná tu `.wav`  
  Los archivos pueden estar en cualquier carpeta.

### Paso 2 — Configurar el espectrograma

| Control | Efecto |
|---|---|
| **Colormap** | Paleta de colores: viridis, plasma, jet, gray (blanco/negro), hot, etc. |
| **Invertir colores** | Invierte la paleta (ej. gray → blanco sobre negro) |
| **Min dB** | Elimina el ruido de fondo (subir = más limpio) |
| **Max dB** | Límite superior de la escala de color |
| **Freq. mín / máx (Hz)** | Filtrá el rango visible (ej. 20000–80000 Hz para USV de ratas) |
| **Ventana FFT** | Mayor = mejor resolución en frecuencia, menor en tiempo |
| **Hop length** | Menor = más resolución temporal |
| **Contraste** | Realza las diferencias de intensidad |
| **Brillo** | Aclara u oscurece la imagen |

Apretá **Ver espectrograma** para previsualizar. Podés ajustar y volver a previsualizar cuantas veces quieras.

### Paso 3 — Generar el video

| Control | Descripción |
|---|---|
| **Archivo de salida** | Ruta del video generado (`.mp4` recomendado) |
| **Tamaño espectrograma** | Porcentaje del ancho del video (10–70%) |
| **Posición** | Dónde se coloca el espectrograma en el frame |
| **Offset audio (s)** | Si el audio y el video no empiezan al mismo tiempo, ajustá aquí (positivo = audio empieza después) |

Apretá **Generar Video**. Aparece una barra de progreso. Cuando termina, el archivo queda guardado en la ruta elegida.

---

## Problemas comunes

| Síntoma | Solución |
|---|---|
| `ModuleNotFoundError` | El entorno virtual no está activado. Ejecutá `.venv\Scripts\Activate.ps1` |
| Video de salida sin imagen | Codec incompatible con el reproductor. Usá `.mp4` como extensión |
| Espectrograma tarda mucho | Usá `Ventana FFT: 1024` y `Hop: 512` para archivos largos |
| La pantalla de preview queda en gris | El espectrograma se está calculando en segundo plano, esperá |
| Video y espectrograma desincronizados | Ajustá el **Offset audio** en Paso 3 |
