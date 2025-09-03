let imageId = null;
let level = 0;
const maxLevel = 20;

const statusEl = document.getElementById('status');
const levelEl = document.getElementById('level');
const previewEl = document.getElementById('preview');
const spinnerEl = document.getElementById('spinner');
const fileInput = document.getElementById('fileInput');
const increaseBtn = document.getElementById('increase');
const decreaseBtn = document.getElementById('decrease');
const resetBtn = document.getElementById('reset');

fileInput.addEventListener('change', async (e) => {
	const file = e.target.files[0];
	if(!file) return;
	statusEl.textContent = 'Yükleniyor...';
	const formData = new FormData();
	formData.append('file', file);
	const res = await fetch('/api/upload', { method: 'POST', body: formData });
	if(!res.ok){ statusEl.textContent = 'Yükleme hatası'; return; }
	const data = await res.json();
	imageId = data.image_id;
	level = 0; levelEl.textContent = String(level);
	await updatePreview();
});

increaseBtn.addEventListener('click', async () => {
	if(imageId === null) return;
	level = Math.min(maxLevel, level + 1);
	levelEl.textContent = String(level);
	await updatePreview();
});

decreaseBtn.addEventListener('click', async () => {
	if(imageId === null) return;
	level = Math.max(-maxLevel, level - 1);
	levelEl.textContent = String(level);
	await updatePreview();
});

resetBtn.addEventListener('click', async () => {
	if(imageId === null) return;
	level = 0; levelEl.textContent = String(level);
	await updatePreview();
});

async function updatePreview(){
	try{
		statusEl.textContent = 'İşleniyor...';
		spinnerEl.classList.add('active');
		previewEl.src = '';
		const res = await fetch(`/api/process?image_id=${imageId}&level=${level}`);
		if(!res.ok){ statusEl.textContent = 'İşleme hatası'; spinnerEl.classList.remove('active'); return; }
		const data = await res.json();
		previewEl.src = `data:image/jpeg;base64,${data.base64}`;
		statusEl.textContent = 'Hazır';
	}catch(err){
		statusEl.textContent = 'Hata';
		console.error(err);
	}finally{
		spinnerEl.classList.remove('active');
	}
} 