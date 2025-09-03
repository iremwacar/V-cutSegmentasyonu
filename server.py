import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pipeline import ImageStore, read_image_to_bgr, process_image_with_weight, encode_bgr_image_to_base64_jpeg

app = FastAPI(title="Vucut Segmentasyonu Web")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def index():
	return """
	<!doctype html>
	<html>
	<head>
		<meta charset='utf-8'/>
		<meta name='viewport' content='width=device-width, initial-scale=1'/>
		<title>Vücut Segmentasyonu</title>
		<style>
			body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;margin:24px;color:#111}
			.container{max-width:960px;margin:0 auto}
			.controls{display:flex;gap:12px;align-items:center;margin-top:12px}
			button{padding:8px 12px;border-radius:8px;border:1px solid #ccc;background:#fff;cursor:pointer}
			button:hover{background:#f5f5f5}
			img{max-width:100%;height:auto;border-radius:12px;border:1px solid #e5e5e5}
			label{display:inline-block;padding:8px 12px;border:1px dashed #888;border-radius:8px;cursor:pointer}
			input[type=file]{display:none}
			.badge{display:inline-block;padding:4px 8px;background:#eef;border:1px solid #ccd;border-radius:999px}
		</style>
	</head>
	<body>
		<div class='container'>
			<h2>Vücut Segmentasyonu Demo</h2>
			<p>Bir fotoğraf yükleyin ve ağırlığı +/- butonları ile ayarlayın.</p>
			<div>
				<label for='fileInput'>Görsel Yükle</label>
				<input id='fileInput' type='file' accept='image/*'/>
				<span id='status' class='badge'>Hazır</span>
			</div>
			<div class='controls'>
				<button id='decrease'>-</button>
				<div>Ağırlık Seviyesi: <strong id='level'>0</strong></div>
				<button id='increase'>+</button>
			</div>
			<div style='margin-top:16px'>
				<img id='preview' alt='Önizleme' />
			</div>
		</div>
		<script>
		let imageId = null;
		let level = 0;
		const maxLevel = 10;
		const statusEl = document.getElementById('status');
		const levelEl = document.getElementById('level');
		const previewEl = document.getElementById('preview');

		document.getElementById('fileInput').addEventListener('change', async (e) => {
			const file = e.target.files[0];
			if(!file) return;
			statusEl.textContent = 'Yükleniyor...';
			const formData = new FormData();
			formData.append('file', file);
			const res = await fetch('/api/upload', { method: 'POST', body: formData });
			if(!res.ok){ statusEl.textContent = 'Yükleme hatası'; return; }
			const data = await res.json();
			imageId = data.image_id;
			level = 0; levelEl.textContent = level;
			await updatePreview();
		});

		document.getElementById('increase').addEventListener('click', async () => {
			if(imageId === null) return;
			level = Math.min(maxLevel, level + 1);
			levelEl.textContent = level;
			await updatePreview();
		});

		document.getElementById('decrease').addEventListener('click', async () => {
			if(imageId === null) return;
			level = Math.max(-maxLevel, level - 1);
			levelEl.textContent = level;
			await updatePreview();
		});

		async function updatePreview(){
			statusEl.textContent = 'İşleniyor...';
			previewEl.src = '';
			const res = await fetch(`/api/process?image_id=${imageId}&level=${level}`);
			if(!res.ok){ statusEl.textContent = 'İşleme hatası'; return; }
			const data = await res.json();
			previewEl.src = `data:image/jpeg;base64,${data.base64}`;
			statusEl.textContent = 'Hazır';
		}
		</script>
		</body>
		</html>
	"""


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
	try:
		content = await file.read()
		image_bgr = read_image_to_bgr(content)
		image_id = ImageStore.put(image_bgr)
		return JSONResponse({"image_id": image_id})
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/process")
async def process(image_id: str, level: int = 0):
	try:
		image_bgr = ImageStore.get(image_id)
		output_bgr = process_image_with_weight(image_bgr, level)
		b64 = encode_bgr_image_to_base64_jpeg(output_bgr)
		return JSONResponse({"base64": b64})
	except KeyError:
		raise HTTPException(status_code=404, detail="Image not found")
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
	uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False) 