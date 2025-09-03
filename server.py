import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import os

from pipeline import ImageStore, read_image_to_bgr, process_image_with_weight, encode_bgr_image_to_base64_jpeg

app = FastAPI(title="Vucut Segmentasyonu Web")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "public")


def _read_public(file_name: str, default: str = "") -> str:
	p = os.path.join(PUBLIC_DIR, file_name)
	if not os.path.exists(p):
		return default
	with open(p, "r", encoding="utf-8") as f:
		return f.read()


@app.get("/", response_class=HTMLResponse)
async def index():
	html = _read_public("index.html", default="<h3>Frontend not built</h3>")
	return HTMLResponse(content=html)


@app.get("/app.css")
async def app_css():
	css = _read_public("app.css", default="")
	return Response(content=css, media_type="text/css")


@app.get("/app.js")
async def app_js():
	js = _read_public("app.js", default="")
	return Response(content=js, media_type="application/javascript")


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
		# Flip sign so that minus => slim (narrow), plus => bulk (widen)
		effective_level = -level
		output_bgr = process_image_with_weight(image_bgr, effective_level)
		b64 = encode_bgr_image_to_base64_jpeg(output_bgr)
		return JSONResponse({"base64": b64})
	except KeyError:
		raise HTTPException(status_code=404, detail="Image not found")
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
	uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False) 