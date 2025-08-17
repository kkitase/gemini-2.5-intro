
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/01-text-generation-and-chat.ipynb)

## STEP 1: Google Colab の環境を準備する

https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started.ipynb から Open in Colab ボタンを押してください。

## STEP 2: SDK をインストールする

次に、Google Colab の環境に Google の Generative AI SDK をインストールします。Google Colab の環境で以下のコマンドを実行するだけです。（すでにコマンドは記載されているので、実行ボタンを押すだけです。）

```python
%pip install -U -q 'google-genai>=1.0.0'
```
![alt text](<../image/ScreenShot 2025-08-13 13.36.30.png>)

## STEP 3: Gemini API キーを入手する

まずは、Gemini API を使うための準備をしましょう。
1.  **Google AI Studio にアクセス**: ウェブブラウザで [Google AI Studio](https://aistudio.google.com/apikey) にアクセスします。
2.  **API キーの取得**: 画面の指示に従って、新しい API キーを作成します。このキーは、アプリケーションから Gemini を呼び出すための「鍵」の役割を果たします。大切に保管してください。

## STEP 4: Gemini API を Google Colab に設定する

1.  Google Colab で左のパネルから、シークレットタブ（🔑）を開きます。
![alt text](<../image/ScreenShot 2025-08-13 13.59.19.png>)
2.  `GEMINI_API_KEY` という名前で、新しいシークレットキーを作成します。
3.  `GEMINI_API_KEY` の「値 (Value)」入力欄に、先ほど作成した Gemini API キーをコピーして貼り付けます。
4.  左側にあるトグルボタンをオンに切り替えて、すべてのノートブックがこのシークレットにアクセスできるように許可します。


