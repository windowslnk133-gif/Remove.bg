import os
import io
import json
import urllib.request
from flask import Flask, request, render_template_string, send_file
import numpy as np
import onnxruntime as ort
from PIL import Image

app = Flask(__name__)

# 免費伺服器專用：線上載入最輕量、最精準的開源去背權重 (U2NETp)
MODEL_URL = "https://github.com"
MODEL_PATH = "u2netp.onnx"

# 下載模型權重
if not os.path.exists(MODEL_PATH):
    print("正在下載輕量化 AI 去背模型...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# 初始化 AI 引擎
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>我的獨立 AI 去背站</title>
    <script src="https://tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen flex flex-col items-center justify-center p-4">
    <div class="bg-white p-8 rounded-2xl shadow-xl max-w-md w-full text-center">
        <h1 class="text-3xl font-bold text-gray-800 mb-2">🤖 獨立自建 AI 去背站</h1>
        <p class="text-gray-500 mb-6">輕量優化版，無次數限制、不看大廠臉色！</p>
        
        <form action="/upload" method="post" enctype="multipart/form-data" class="space-y-4">
            <div class="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-xl p-6 bg-gray-50 relative cursor-pointer hover:bg-gray-100">
                <input type="file" name="file" accept="image/*" required class="absolute inset-0 opacity-0 cursor-pointer" onchange="this.form.submit()">
                <svg class="h-12 w-12 text-gray-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                <span class="text-gray-600 font-medium">點擊或拖曳圖片至此開始去背</span>
            </div>
        </form>
        <p class="text-xs text-gray-400 mt-4">支援 JPG, PNG, WebP，自動輸出高畫質透明圖</p>
    </div>
</body>
</html>
"""

def process_img(img):
    # 圖片預處理符合 AI 輸入
    img_gray = img.convert('RGB').resize((320, 320))
    img_np = np.array(img_gray).astype(np.float32) / 255.0
    tmpImg = np.zeros((320, 320, 3))
    tmpImg[:,:,0] = (img_np[:,:,0] - 0.485) / 0.229
    tmpImg[:,:,1] = (img_np[:,:,1] - 0.456) / 0.224
    tmpImg[:,:,2] = (img_np[:,:,2] - 0.406) / 0.225
    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = np.expand_dims(tmpImg, list(range(1, 1 + 4 - len(tmpImg.shape))))
    tmpImg = tmpImg.astype(np.float32)
    
    # 執行 AI 推理
    inputs = {session.get_inputs()[0].name: tmpImg}
    pred = session.run(None, inputs)[0][0][0]
    
    # 後處理：將遮罩還原回原圖大小並混合
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    mask = Image.fromarray((pred * 255).astype(np.uint8)).resize(img.size, resample=Image.BILINEAR)
    
    empty = Image.new("RGBA", img.size, (0, 0, 0, 0))
    return Image.composite(img.convert("RGBA"), empty, mask)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "未上傳檔案", 400
    file = request.files['file']
    if file.filename == '':
        return "未選擇檔案", 400

    try:
        input_image = Image.open(file.stream)
        output_image = process_img(input_image)

        img_io = io.BytesIO()
        output_image.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='ai_removed.png')
    except Exception as e:
        return f"去背出錯: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
