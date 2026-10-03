import os
from flask import Flask, request, render_template_string, send_file
from rembg import remove, new_session
import io
from PIL import Image

app = Flask(__name__)

# 初始化去背模型（使用開源最強的 rmbg 權重）
session = new_session("rmbg")

# 精美的 HTML/CSS 網頁介面
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
        <p class="text-gray-500 mb-6">100% 國產自製，無次數限制、不對外連線！</p>
        
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
        # 讀取圖片並用本地 AI 模型進行去背
        input_image = Image.open(file.stream)
        output_image = remove(input_image, session=session)

        # 將結果存入記憶體並回傳讓使用者下載
        img_io = io.BytesIO()
        output_image.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='ai_removed.png')
    except Exception as e:
        return f"去背出錯: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
