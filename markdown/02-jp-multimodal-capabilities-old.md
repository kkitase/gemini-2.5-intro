[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/02-multimodal-capabilities.ipynb)

# パート 2 - マルチモーダル機能 (画像、音声、動画、ドキュメント)

`gemini-2.5-flash` のような Gemini モデルは、`client.models.generate_content()` を使用して、単一のプロンプトでテキスト、画像、音声、動画、ドキュメントを処理できます。これにより、さまざまなメディアタイプのコンテンツを理解し、生成できる強力なマルチモーダル AI アプリケーションを開発できます。

**主な機能:**
- **視覚的理解**: 画像の分析、テキストの抽出、オブジェクトの識別
- **音声処理**: 音声の文字起こし、音楽の分析、音声コンテンツの理解
- **動画分析**: 動画の要約、キーフレームの抽出、動きの理解
- **ドキュメント処理**: PDF からの情報の抽出、レイアウトの理解
- **マルチモーダル生成**: テキストプロンプトからの画像と音声の作成


```python
# 画像処理ライブラリPillowをインストール
# %pip install pillow
```


```python
from google import genai
from google.genai import types

# 画像処理のためのライブラリをインポート
from PIL import Image

# メモリ上でバイナリデータを扱うためのライブラリをインポート
from io import BytesIO

# Google Colab でのユーザーデータを利用するためのライブラリをインポート
from google.colab import userdata
GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')

# APIキーでクライアントを作成
MODEL_ID = "gemini-2.5-flash"
client = genai.Client(api_key=GEMINI_API_KEY)
```

## 1. 画像理解: 単一の画像

Gemini は、PIL `Image` オブジェクト、生のバイト、または File API を介してアップロードされたファイルなど、複数の形式の画像を分析できます。

**各メソッドの使い分け:**
- **生のバイト**: API やメモリからの画像データを扱う場合
- **File API**: 20MB を超える大きな画像や、複数のリクエストで画像を再利用したい場合


```python
    
```
