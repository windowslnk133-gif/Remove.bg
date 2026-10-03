import io
import os
import urllib.request
from flask import Flask, render_template_string, request, send_file
import numpy as np
import onnxruntime as ort
from PIL import Image

app = Flask(__name__)

# 載入輕量化 AI 去背模型 (U2NETp)
MODEL_URL = "https://huggingface.co/danielgatis/rembg/resolve/main/u2netp.onnx"
MODEL_PATH = "u2netp.onnx"

if not os.path.exists(MODEL_PATH):
    print("正在下載輕量化 AI 去背模型...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("模型下載完成！")
    except Exception as e:
        print(f"下載模型失敗: {e}")

try:
    session = ort.InferenceSession(
        MODEL_PATH, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    print("AI 去背引擎初始化成功！")
except Exception as e:
    print(f"AI 引擎初始化失敗: {e}")
    session = None

# 超美現代化網頁 UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Magic Cut - 智慧去背大師</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .transparent-bg {
            background-image: linear-gradient(45deg, #e0e0e0 25%, transparent 25%), linear-gradient(-45deg, #e0e0e0 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #e0e0e0 75%), linear-gradient(-45deg, transparent 75%, #e0e0e0 75%);
            background-size: 20px 20px;
            background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
            background-color: #f3f4f6;
        }
    </style>
</head>
<body class="bg-gradient-to-br from-indigo-950 via-slate-900 to-blue-950 min-h-screen text-slate-100 font-sans flex flex-col justify-between antialiased selection:bg-indigo-500 selection:text-white">

    <div class="h-1.5 w-full bg-gradient-to-r from-pink-500 via-purple-500 to-indigo-500"></div>

    <main class="container mx-auto px-4 py-8 flex-grow flex items-center justify-center max-w-5xl">
        <div class="w-full bg-slate-900/60 backdrop-blur-xl border border-slate-800 p-6 md:p-10 rounded-3xl shadow-2xl transition-all duration-300">
            
            <div class="text-center mb-8">
                <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-semibold tracking-wide uppercase mb-3 animate-pulse">
                    <i class="fa-solid fa-sparkles"></i> AI Engine v1.4 Powered
                </div>
                <h1 class="text-4xl md:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-200 to-indigo-300 bg-clip-text text-transparent">
                    AI Magic Cut
                </h1>
                <p class="text-slate-400 mt-2 text-sm md:text-base">
                    極致美觀、速度飛快，完全免費的獨立相片去背站
                </p>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
                
                <div class="space-y-4">
                    <h3 class="text-sm font-semibold text-slate-400 tracking-wider uppercase flex items-center gap-2">
                        <i class="fa-solid fa-image text-indigo-400"></i> 原始圖片
                    </h3>
                    <form id="upload-form" action="/upload" method="post" enctype="multipart/form-data">
                        <div id="drop-zone" class="group relative border-2 border-dashed border-slate-700 hover:border-indigo-500 rounded-2xl p-8 bg-slate-950/40 hover:bg-slate-950/80 transition-all duration-300 min-h-[320px] flex flex-col items-center justify-center overflow-hidden">
                            <input type="file" id="file-input" name="file" accept="image/*" required class="absolute inset-0 opacity-0 cursor-pointer z-10">
                            
                            <div id="upload-prompt" class="text-center space-y-4 transition-all duration-300 group-hover:scale-105">
                                <div class="w-16 h-16 mx-auto rounded-2xl bg-slate-800 flex items-center justify-center text-slate-400 group-hover:text-indigo-400 group-hover:bg-indigo-500/10 transition-all duration-300 shadow-inner">
                                    <i class="fa-solid fa-cloud-arrow-up text-2xl"></i>
                                </div>
                                <div>
                                    <p class="text-slate-200 font-medium">點擊或拖曳檔案至此</p>
                                    <p class="text-xs text-slate-500 mt-1">支援 JPG, PNG, WebP 高畫質影像</p>
                                </div>
                            </div>

                            <img id="input-preview" class="max-h-[300px] w-full object-contain hidden rounded-xl shadow-lg relative z-0" />
                        </div>
                    </form>
                </div>

                <div class="space-y-4">
                    <h3 class="text-sm font-semibold text-slate-400 tracking-wider uppercase flex items-center gap-2">
                        <i class="fa-solid fa-wand-magic-sparkles text-purple-400"></i> 去背成果
                    </h3>
                    <div class="transparent-bg border border-slate-800 rounded-2xl min-h-[320px] flex flex-col items-center justify-center relative overflow-hidden group shadow-inner">
                        
                        <div id="output-prompt" class="text-slate-400 text-center space-y-2 p-6">
                            <i class="fa-solid fa-layer-group text-3xl opacity-20 block mb-2"></i>
                            <p class="text-sm">等待左側上傳圖片...</p>
                        </div>

                        <div id="loading-spinner" class="hidden text-center space-y-4">
                            <div class="relative w-12 h-12 mx-auto">
                                <div class="absolute inset-0 rounded-full border-4 border-slate-800"></div>
                                <div class="absolute inset-0 rounded-full border-4 border-t-indigo-500 animate-spin"></div>
                            </div>
                            <p class="text-xs text-indigo-400 font-medium tracking-wide animate-pulse">AI 正在邊緣精密計算中...</p>
                        </div>

                        <img id="output-preview" class="max-h-[300px] w-full object-contain hidden rounded-xl p-2 select-none group-hover:scale-[1.02] transition-transform duration-300" />
                    </div>
                </div>

            </div>

            <div id="action-area" class="mt-8 pt-6 border-t border-slate-800/60 hidden text-center">
                <button id="download-btn" class="inline-flex items-center gap-2 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white font-bold py-3.5 px-8 rounded-xl shadow-lg hover:shadow-indigo-500/20 active:scale-[0.98] transition-all duration-200">
                    <i class="fa-solid fa-download"></i> 下載高畫質透明 PNG ✨
                </button>
                <button onclick="window.location.reload()" class="ml-4 inline-flex items-center gap-2 bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium py-3.5 px-5 rounded-xl border border-slate-700 transition-all duration-200 text-sm">
                    <i class="fa-solid fa-arrow-rotate-left"></i> 重新一張
                </button>
            </div>

        </div>
    </main>

    <footer class="text-center py-4 text-xs text-slate-600 border-t border-slate-900">
        &copy; 2026 AI Magic Cut. 全本機隱私運算，永久免費。
    </footer>

    <script>
        const fileInput = document.getElementById('file-input');
        const inputPreview = document.getElementById('input-preview');
        const outputPreview = document.getElementById('output-preview');
        const uploadPrompt = document.getElementById('upload-prompt');
        const outputPrompt = document.getElementById('output-prompt');
        const loadingSpinner = document.getElementById('loading-spinner');
        const actionArea = document.getElementById('action-area');
        const downloadBtn = document.getElementById('download-btn');
        const dropZone = document.getElementById('drop-zone');

        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            inputPreview.src = URL.createObjectURL(file);
            inputPreview.classList.remove('hidden');
            uploadPrompt.classList.add('hidden');
            dropZone.classList.remove('border-dashed');
            dropZone.classList.add('border-slate-800');

            outputPrompt.classList.add('hidden');
            loadingSpinner.classList.remove('hidden');
            outputPreview.classList.add('hidden');
            actionArea.classList.add('hidden');

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const downloadUrl = URL.createObjectURL(blob);

                    loadingSpinner.classList.add('hidden');
                    outputPreview.src = downloadUrl;
                    outputPreview.classList.remove('hidden');

                    actionArea.classList.remove('hidden');
                    downloadBtn.onclick = () => {
                        const a = document.createElement('a');
                        a.href = downloadUrl;
                        a.download = `magic-cut-${Date.now()}.png`;
                        a.click();
                    };
                } else {
                    alert('後端 AI 運算發生錯誤。');
                    window.location.reload();
                }
            } catch (error) {
                alert('網路連線失敗！');
                window.location.reload();
            }
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.add('border-indigo-500', 'bg-indigo-500/5'), false);
        });
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.remove('border-indigo-500', 'bg-indigo-500/5'), false);
        });
    </script>
</body>
</html>
"""


def process_img(img):
    if session is None:
        raise Exception("AI 引擎未成功載入")

    img_gray = img.convert("RGB").resize((320, 320))
    img_np = np.array(img_gray).astype(np.float32) / 255.0

    tmpImg = np.zeros((320, 320, 3))
    tmpImg[:, :, 0] = (img_np[:, :, 0] - 0.485) / 0.229
    tmpImg[:, :, 1] = (img_np[:, :, 1] - 0.456) / 0.224
    tmpImg[:, :, 2] = (img_np[:, :, 2] - 0.406) / 0.225

    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = np.expand_dims(tmpImg, axis=0).astype(np.float32)

    inputs = {input_name: tmpImg}
    pred = session.run(None, inputs)[0]

    pred = (pred - pred.min()) / (pred.max() - pred.min())
    pred = pred[0][0]

    mask = Image.fromarray((pred * 255).astype(np.uint8)).resize(
        img.size, resample=Image.BILINEAR
    )
    empty = Image.new("RGBA", img.size, (0, 0, 0, 0))

    return Image.composite(img.convert("RGBA"), empty, mask)


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file provided", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    try:
        input_image = Image.open(file.stream)
        output_image = process_img(input_image)

        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        return send_file(img_byte_arr, mimetype="image/png")
    except Exception as e:
        print(f"處理圖片失敗: {e}")
        return str(e), 500


if __name__ == "__main__":
    print("啟動 AI 去背 Web App...")
    app.run(host="0.0.0.0", port=5000, debug=True)
