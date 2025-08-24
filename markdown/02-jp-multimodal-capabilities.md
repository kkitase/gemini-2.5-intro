# Google Colab ではじめる Gemini API、マルチモーダル機能 (画像、音声、動画、ドキュメント)を試してみよう。

`gemini-2.5-flash` のような Gemini モデルは、`client.models.generate_content()` を使用して、単一のプロンプトでテキスト、画像、音声、動画、ドキュメントを処理できます。これにより、さまざまなメディアタイプのコンテンツを理解し、生成できる強力なマルチモーダル AI アプリケーションを開発できます。

**主な機能:**
- **視覚的理解**: 画像の分析、テキストの抽出、オブジェクトの識別
- **音声処理**: 音声の文字起こし、音楽の分析、音声コンテンツの理解
- **動画分析**: 動画の要約、キーフレームの抽出、動きの理解
- **ドキュメント処理**: PDF からの情報の抽出、レイアウトの理解
- **マルチモーダル生成**: テキストプロンプトからの画像と音声の作成

以降の解説は、Google Colab で実際にコードを実行しながら進めることを想定していますが、コードと解説を読み進めるだけでも学習できます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kkitase/gemini-2.5-findy/blob/main/notebooks/02-jp-multimodal-capabilities.ipynb)

## 重要: 環境の準備
- [セットアップと認証](https://colab.research.google.com/github/kkitase/gemini-2.5-findy/blob/main/notebooks/00-jp-setup-and-authentication.ipynb#scrollTo=bfd5d261) のセクションを完了していることを確認してください。
- もしエラーが出たら、[Gemini in Google Colab](https://colab.research.google.com/github/kkitase/gemini-2.5-findy/blob/main/notebooks/00-jp-setup-and-authentication.ipynb#scrollTo=7d140654) を使い、コードの説明やデバッグをして解決を試みてください。


### 学習のためのリポジトリをクローン
```python
!git clone https://github.com/kkitase/gemini-2.5-findy.git
%cd gemini-2.5-findy
```

## 1. 画像理解: 単一の画像

Gemini は、PIL `Image` オブジェクト、画像のローデータ、または File API を使ってアップロードされたファイルなど、複数の形式の画像を分析できます。

**各メソッドの使い分け:**
- **画像のローデータ**: API やメモリからの画像データを扱う場合
- **File API**: 20MB を超える大きな画像や、複数のリクエストで画像を再利用したい場合


### 画像処理を行う準備
画像処理を行うために必要なツールやライブラリのインストールなど、準備をします。

```python
# 画像処理ライブラリPillowをインストール
%pip install pillow
```


```python
# Gemini APIとHTTPリクエスト用のライブラリをインポート
from google import genai
from google.genai import types
import requests

# 画像処理のためのライブラリをインポート
from PIL import Image

# メモリ上でバイナリデータを扱うためのライブラリをインポート
from io import BytesIO

# Google Colabでのユーザーデータを利用するためのライブラリをインポート
from google.colab import userdata
GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')

# APIキーでクライアントを作成
MODEL_ID = "gemini-2.5-flash" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"] {allow-input: true}
client = genai.Client(api_key=GEMINI_API_KEY)
```


```python
# サンプル画像をダウンロードします
!curl -o image.jpg "https://storage.googleapis.com/generativeai-downloads/images/Cupcakes.jpg"
```

### バイナリデータを読み込んで画像解析
```python
# 画像ファイルをバイナリモードで開き、その内容を読み込みます
with open('image.jpg', 'rb') as f:
    image_bytes = f.read()

# テキストプロンプトと画像データをモデルに送信して、コンテンツを生成します
response_specific = client.models.generate_content(
    model=MODEL_ID,
    contents=["これは何の画像ですか？", 
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")]
)
# 生成されたテキストの応答を出力します
print(response_specific.text)
```

### File API を使って画像解析
20MB を超える大きなペイロードには File API を使用できます。


```python
# ファイルを File API にアップロードします
file_id = client.files.upload(file="assets/data/Cupcakes.jpg")

# アップロードしたファイルとプロンプトを送信して、コンテンツを生成します
response = client.models.generate_content(
    model=MODEL_ID,
    contents=["これは何の画像ですか？", file_id]
)

# 生成されたテキストの応答を出力します
print(response.text)
```

> File API を使用すると、プロジェクトごとに最大 20 GB のファイルを保存でき、ファイルごとの最大サイズは 2 GB です。ファイルは 48 時間保存されます。その期間中は API キーでアクセスできますが、API からダウンロードすることはできません。Gemini API が利用可能なすべてのリージョンで無料で利用できます。


## 2. 画像理解: 複数の画像

Gemini は複数の画像を同時に分析および比較できます。これは、比較分析、視覚的なストーリーテリング、または一連の出来事の理解に強力です。


```python
# 比較したい画像のURLを定義します
image_url_1 = "https://plus.unsplash.com/premium_photo-1694819488591-a43907d1c5cc?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8Y3V0ZSUyMGRvZ3xlbnwwfHwwfHx8MA%3D%3D" # 犬
image_url_2 = "https://images.pexels.com/photos/2071882/pexels-photo-2071882.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500" # 猫

# requestsライブラリを使って、URLから画像データを取得します
image_response_req_1 = requests.get(image_url_1)
image_response_req_2 = requests.get(image_url_2)

response_multi = client.models.generate_content(
     model=MODEL_ID,
     contents=[
         "この2つの画像を比較してください。それぞれの主な被写体は何で、何をしていますか？",
         "画像 1:",
         types.Part.from_bytes(data=image_response_req_1.content, mime_type="image/jpeg"),
         "画像 2:",
         types.Part.from_bytes(data=image_response_req_2.content, mime_type="image/jpeg")
     ]
 )
print(response_multi.text)
```

## 3. 音声解析

Gemini は、文字起こし、音声コンテンツ解析、話者識別、音声要約のために音声ファイルを処理できます。これは、ポッドキャスト、会議、インタビュー、ボイスメモに特に役立ちます。

**サポートされている音声形式**: MP3、WAV、FLAC、AAC、およびその他の一般的な形式


```python
file_path = "assets/data/audio2.mp3"

file_id = client.files.upload(file=file_path)

# Gemini APIを使用して構造化された応答を生成する
prompt = """エピソードのスクリプトを生成してください。タイムスタンプを含め、話者を特定してください。

話者:
- Speaker 1: セラフィーナ・クローデル（セラフィーナ）
- Speaker 2: 天堂 健（けん）

例:
[00:00] セラフィーナ: こんにちは。
[00:02] けん: こんにちは。

正しい話者名を含めることが重要です。前に特定した名前を使用してください。話者の名前が本当にわからない場合は、アルファベットの文字で特定してください。たとえば、不明な話者「A」と別の不明な話者「B」がいる場合があります。

音楽または短いジングルが再生されている場合は、次のように示してください。
[01:02] [MUSIC] または [01:02] [JINGLE]

再生されている音楽またはジングルの名前を特定できる場合は、代わりにそれを使用してください。例:
[01:02] [Firework by Katy Perry] または [01:02] [The Sofa Shop jingle]

他の音が再生されている場合は、その音を特定してみてください。例:
[01:02] [Bell ringing]

個々のキャプションはそれぞれ非常に短く、最大でも数文程度にしてください。

エピソードの終わりを [END] で示してください。

太字や斜体などのマークダウン書式は使用しないでください。

外国の文字が正しいと確信している場合を除き、英字のみを使用してください。

正しい単語を使用し、すべてを正しく綴ることが重要です。ポッドキャストのコンテキストを参考にしてください。
ホストが映画、本、有名人などについて話す場合は、映画、本、有名人の名前が正しく綴られていることを確認してください。"""
audio_part = types.Part.from_uri(file_uri=file_id.uri, mime_type=file_id.mime_type)

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, audio_part]
)
print(response.text)
```

## 4. 動画解析

Gemini は動画ファイルを処理して、その内容を理解し、シーンを解析し、オブジェクトとアクションを特定し、詳細な要約を提供できます。

**動画機能:**
- シーンの分析と要約
- オブジェクトとアクションの認識
- 時間的理解 (いつ何が起こるか)
- コンテンツの抽出と重要な瞬間
- YouTube 動画の分析

### 保存された動画の解析
```python
# 動画を表示します
from IPython.display import Video
video_path = "assets/data/dear.mp4"
Video(video_path, embed=True, width=400)
```

```python
# 処理を一時停止するための sleep 関数をインポートします
from time import sleep

# 分析したい動画ファイルのパスを指定します
video_path = "assets/data/dear.mp4"

# 動画ファイルを File API にアップロードします
video_file_id = client.files.upload(file=video_path)

# ファイルの処理が完了するまで待機する関数を定義します
def wait_for_file_ready(file_id):
    # ファイルの状態が「PROCESSING」である間、ループを続けます
    while file_id.state == "PROCESSING":
        sleep(1) # 1秒間待機します
        # ファイルの最新の状態を取得します
        file_id = client.files.get(name=file_id.name)
    return file_id

# ファイルの準備ができるまで待機します
video_file_id = wait_for_file_ready(video_file_id)

# 動画に関するプロンプトを作成します
prompt = "この動画について説明してください。"

# 準備ができたファイルから、APIに渡すためのPartオブジェクトを作成します
video_part = types.Part.from_uri(file_uri=video_file_id.uri, mime_type=video_file_id.mime_type)

# モデルにプロンプトと動画を送信して、コンテンツを生成します
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, video_part]
)

# 生成されたテキストの応答を出力します
print(response.text)
```

### YouTube 動画の分析

Gemini API は直接の YouTube URL 分析をサポートしていて、動画コンテンツ分析に非常に便利です。


```python
# YouTube 動画を確認
# YouTube動画をノートブックに埋め込むためのライブラリをインポート
from IPython.display import YouTubeVideo

# 表示したいYouTube動画のURLを定義
youtube_url = "https://www.youtube.com/watch?v=CN_a-uSK67s"

# URLから動画IDを抽出
video_id = youtube_url.split("v=")[1]

# 動画IDを使って、動画を埋め込み表示
YouTubeVideo(video_id)
```


```python
# YouTune 動画を解析
# 分析したいYouTube動画のURLを定義
youtube_url = "https://www.youtube.com/watch?v=CN_a-uSK67s"

# URLからAPIに渡すためのPartオブジェクトを作成します
youtube_part = genai.types.Part(
    file_data=genai.types.FileData(file_uri=youtube_url)
)
# 動画に関する具体的な質問をプロンプトとして作成
prompt = """この動画について、200 字でまとめて。
また、動画の中で湯呑みの画像が出てくるタイムスタンプを教えて。"""

# モデルにプロンプトと動画を送信して、コンテンツを生成
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, youtube_part]
)

# 生成されたテキストの応答を出力します
print(response.text)
```

## 5. PDF/ドキュメントファイルの操作

Gemini は PDF などドキュメントの情報を抽出できるため、ドキュメント分析、データ抽出、コンテンツ要約に優れています。

**一般的な使用例:**
- 請求書の処理とデータ抽出
- 契約書の分析と要約
- 研究論文の分析
- フォームの処理と検証
- ドキュメントの分類とルーティング


```python
# 分析したいPDFファイルのパスを指定
pdf_file_path = "assets/data/komeda.pdf"

# PDFファイルをFile APIにアップロード
pdf_file_id = client.files.upload(file=pdf_file_path)

# PDFに関する質問をプロンプトとして作成
prompt = "支払総額はいくらですか？"

# アップロードしたファイルから、APIに渡すためのPartオブジェクトを作成
pdf_part = types.Part.from_uri(file_uri=pdf_file_id.uri, mime_type=pdf_file_id.mime_type)

# モデルにプロンプトとPDFを送信して、コンテンツを生成
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, pdf_part]
)
print(response.text)
```

## 6. コード

Gemini はコードの理解と生成にも大変優れています。[gitingest](https://github.com/cyclotruc/gitingest) を使用して GitHub リポジトリとチャットしてみましょう。gitingest は、GitHub リポジトリのデータを抽出し、LLM が理解できるようなデータに変換するためのPythonスクリプトです。


```python
%pip install gitingest
```


```python
from gitingest import ingest_async

summary, tree, content = await ingest_async("https://github.com/google-gemini/veo-3-gemini-api-quickstart")
```


```python
print(summary)
```


```python
print(tree)
```


```python
prompt = f"""このリポジトリが何についてのものか説明してください:

コード:
{content}
"""

chat = client.chats.create(model=MODEL_ID)

response = chat.send_message(prompt)
print(response.text)
```


```python
response = chat.send_message("スキーマはどのように定義されていますか？")
print(response.text)
```


```python
response = chat.send_message("""すべてのスキーマルートを更新して、Imagen 4 通常モデル `imagen-4.0-generate-001` を使用するようにしてください。
コメントも日本語で追加してください。
更新されたファイルのみを返してください。""")
print(response.text)
```

## 7. 画像生成

Gemini の画像生成機能を使用して、高品質の画像を生成します。この機能は、ビジュアルコンテンツ、プロトタイプ、マーケティング資料、クリエイティブプロジェクトの作成に最適です。

**画像生成機能:**
- テキストから画像への生成
- プロンプトによるスタイルの制御
- 高解像度出力
- 信頼性のための SynthID ウォーターマーク
- 複数のアスペクト比とサイズ


```python
# 画像処理のためのライブラリをインポート
from PIL import Image
# メモリ上でバイナリデータを扱うためのライブラリをインポート
from io import BytesIO

# 画像生成のためのプロンプトを定義
prompt_text = "猫の写真"

# 画像生成モデルを呼び出して、コンテンツを生成
response = client.models.generate_content(
    model="gemini-2.0-flash-preview-image-generation", # 画像生成専用のモデルID
    contents=prompt_text,
    config=types.GenerateContentConfig(
      # 応答にテキストと画像の両方を含めるように指定します
      response_modalities=['TEXT', 'IMAGE']
    )
)

# 応答には複数のパート（テキストや画像など）が含まれている可能性があるため、ループで処理します
for part in response.candidates[0].content.parts:
  # パートがテキストの場合
  if part.text is not None:
    print(f"テキスト応答: {part.text}")
  # パートが画像データの場合
  elif part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
      # 画像データを読み込んで開きます
      image = Image.open(BytesIO(part.inline_data.data))
      # 保存するファイル名を定義します
      image_filename = 'gemini_generated_image.png'
      # 画像をファイルとして保存します
      image.save(image_filename)

# 生成された画像を表示します
image
```

**画像生成のヒント:**
- スタイル (写実的、イラスト、漫画など) を具体的に指定する
- 照明と雰囲気の記述子を含める
- 構図の詳細 (クローズアップ、ワイドショットなど) を指定する
- 関連する場合は、アートスタイルや参照に言及する
- アスペクト比と解像度のニーズを考慮する

> **注**: 生成されたすべての画像には、信頼性検証のために SynthID ウォーターマークが含まれています。詳細は[公式ドキュメント](https://ai.google.dev/gemini-api/docs/image-generation) をご覧ください。


## 8. テキスト読み上げ

テキストを自然な音声に変換します。この機能により、音声コンテンツ、アクセシビリティ機能、インタラクティブなアプリケーションの作成が可能になります。例えば、オーディオブックや、ニュース記事の読み上げ、また、どんな人でも、情報やサービスを同じように利用しやすくするようなアプリケーションを作成できます。

**TTS 機能:**
- 複数の音声オプションとスタイル
- コントロール可能なペース、トーン、感情
- 単一話者および複数話者の音声
- 高品質の音声出力
- 自然言語による音声指示

この例では、`gemini-2.5-flash-preview-tts` モデルを使用して単一話者の音声を生成します。`response_modalities` を `["AUDIO"]` に設定し、`SpeechConfig` を提供する必要があります。


```python
# 音声データを扱うためのライブラリをインストールします
%pip install soundfile numpy
```


```python
# soundfile: WAVファイルなどの音声データを書き込むためのライブラリ
# numpy: 音声データの数値配列を効率的に扱うためのライブラリ
# IPython.display.Audio, display: ノートブック上で音声を再生するためのライブラリ
import soundfile as sf
import numpy as np
from IPython.display import Audio, display

# 音声に変換したいテキストを定義します。「ゆっくり不気味に言ってください:」の部分は、モデルに声のトーンを指示するプロンプトです
text_to_speak = """ゆっくり不気味に言ってください:
プログラマの天堂健（35）。彼の生きがいは、激務の果てに訪れるサウナでの究極の心身解放――「ととのい」にあった。
ある夜、半年がかりのプロジェクトを完遂させた彼は、聖地と崇めるサウナへ向かう。
灼熱のサウナ、極冷の水風呂。そして外気浴でリクライニングチェアに身を横たえ、夜空を見上げた瞬間、健の意識は過去最高の多幸感と共に突き抜けた。
それが、地球での最後の記憶。
次に目覚めた時、そこは鬱蒼とした異世界の森の中だった。
"""

# テキスト読み上げ（TTS）専用のモデルを呼び出します
# 利用可能な声の一覧から 'Kore' を選択します
selected_voice = types.PrebuiltVoiceConfig(
    voice_name='Kore'
)

# 声の詳細設定を作成
# 上で選んだ声を voice_config として設定
voice_settings = types.VoiceConfig(
    prebuilt_voice_config=selected_voice
)

# 上の声の詳細設定を speech_config として設定
speech_settings = types.SpeechConfig(
    voice_config=voice_settings
)

# 応答として「音声(AUDIO)」をリクエストし、スピーチ設定を適用
final_config = types.GenerateContentConfig(
    response_modalities=["AUDIO"],
    speech_config=speech_settings,
)

# TTS専用モデルに、話す内容(text_to_speak)と最終的な設定(final_config)を渡す。
response_tts = client.models.generate_content(
   model="gemini-2.5-flash-preview-tts",
   contents=text_to_speak,
   config=final_config,
)

# 返ってきた音声データ（バイト列）を、16ビット整数のNumPy配列に変換します
audio_array = np.frombuffer(response_tts.candidates[0].content.parts[0].inline_data.data, dtype=np.int16)

# NumPy配列を、サンプルレート24000のWAVファイルとして書き出します
sf.write("generated_speech.wav", audio_array, 24000)

# ノートブック上で再生できるように、音声プレイヤーを表示します
display(Audio("generated_speech.wav"))
```
