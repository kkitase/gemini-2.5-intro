# Google Colab ではじめる Gemini API、テキスト生成やチャットを試してみよう。

ここでは、Google Gen AI SDK を使用した Gemini API によるテキストの生成について、ハンズオン形式で学びます。具体的には、基本的なプロンプト、チャットの対話、ストリーミング、設定について解説します。

以下のボタンから Notebook を開いて進めましょう。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kkitase/gemini-2.5-findy/blob/main/notebooks/01-jp-text-generation-and-chat.ipynb)

以降の解説は、Google Colab で実際にコードを実行しながら進めることを想定していますが、コードと解説を読み進めるだけでも学習できます。

## 重要: 環境の準備
- [セットアップと認証](https://colab.research.google.com/github/kkitase/gemini-2.5-findy/blob/main/notebooks/00-jp-setup-and-authentication.ipynb#scrollTo=bfd5d261) のセクションを完了していることを確認してください。
- もしエラーが出たら、[Gemini in Google Colab](https://colab.research.google.com/github/kkitase/gemini-2.5-findy/blob/main/notebooks/00-jp-setup-and-authentication.ipynb#scrollTo=7d140654) を使い、コードの説明やデバッグをして解決を試みてください。

## 1. 最初のプロンプト

まず、Gemini モデルと対話するためにノートブック全体で使用するクライアントオブジェクトを初期化します。


```python
# Gemini API を Pythonで利用するためのライブラリをインポート
# これにより、テキスト生成、翻訳、要約などの機能を利用できます。
from google import genai

# Google Colab でのユーザーデータを利用するためのライブラリをインポート
from google.colab import userdata

# Google Colab のユーザーデータから API キーを取得
GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')

# API キーを使ってクライアントを作成
client = genai.Client(api_key=GEMINI_API_KEY)

# モデル ID を設定
MODEL_ID = "gemini-2.5-flash" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"] {allow-input: true}
```

次に、プロンプトを作成して、テキストを生成します。


```python
prompt = "都会に住む人をターゲットにした新しいサウナの名前を 3 つ作成してください。"

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)

print("Gemini からの応答:")
print(response.text)
```


```python
# TODO: 
# さまざまなプロンプトを送信してみましょう。
#
```

## 2. トークンの理解とカウント

トークンは、Gemini モデルがテキストを処理するために使用する基本的な単位です。トークンの使用状況を理解することは、次の点で重要です。
- **コスト管理**: 請求はトークン消費量に基づいています
- **コンテキスト制限**: モデルには最大トークン制限があります（例：Gemini 2.5 Pro の場合は 100 万トークン）
- **パフォーマンスの最適化**: 入力が小さいほど、一般的に処理が高速になります

Gemini モデルの場合、1 トークンは約英数字 4 文字に相当し、100 トークンは約 60 〜 80 の英単語に相当します。

### 送信前のトークンをカウント

モデルに送信する前に入力のトークンをカウントして、コストを見積もり、制限内に収まるようにすることができます。


```python
# 入力のトークンをカウントする
prompt = "あなたはサウナについて専門知識を持っています。サウナで、ととのうとはどういうことですか？"

# TODO: 
# client.models.count_tokens() メソッドを呼び出します。下記のコードを修正して、MODEL_ID と prompt を渡すようにしてください。
# token_count = client.models.count_tokens(
#     model=...,
#     contents=...
# )
print(f"入力トークン: {token_count.total_tokens}")

# コストの見積もり（価格例 - 現在のレートを確認してください）
estimated_cost = token_count.total_tokens * 0.15 / 1_000_000
print(f"推定入力コスト: ${estimated_cost:.6f}")
```

### 送信後のトークンをカウント

テキストを生成した後、詳細なトークン使用状況情報にアクセスできます。


```python
prompt = "サウナでととのった後に飲む飲み物を 3 つ提案して。"

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)

print(f"提案:\n{response.text}\n")

# トークン使用メタデータにアクセスする
usage = response.usage_metadata
print(f"入力トークン: {usage.prompt_token_count}")
print(f"思考トークン: {usage.thoughts_token_count}")
print(f"出力トークン: {usage.candidates_token_count}")

# 合計推定コストを計算する
total_cost = (usage.prompt_token_count * 0.15 + (usage.candidates_token_count + usage.thoughts_token_count) * 3.5) / 1_000_000
print(f"合計推定コスト: ${total_cost:.6f}")
```

## 3. テキスト生成の基本

テキストを生成する最も簡単な方法は、モデルにテキストのみのプロンプトを提供することです。 `contents` は、単一のプロンプト、リスト、またはマルチモーダル入力の組み合わせにすることができます。


```python
response_sauna_origin = client.models.generate_content(
    model=MODEL_ID,
    contents="サウナの発祥はどこですか？"
)
print(f"質問: サウナの発祥はどこですか？\nA: {response_sauna_origin.text}")
```


```python
response_gnocchi_tokyo = client.models.generate_content(
    model=MODEL_ID,
    contents=["新しいニョッキ専門レストランの名前を考えて", "city: tokyo"]
)
print(f"\n東京のニョッキ レストラン:\n{response_gnocchi_tokyo.text}")
```


```python
# TODO:
# さまざまなテキストプロンプトを送信してみましょう。
#
```

## 4. ストリーミング

ストリーミングを使用すると、生成される応答を少しずつ受け取ることできるため、長い応答やチャットボットなどのリアルタイム アプリケーションで、より優れたユーザー体験を提供できます。

**ストリーミングを使用する場合:**
- インタラクティブなアプリケーション（チャットボットなど）
- 長いコンテンツの生成
- ユーザー体験の向上


```python
prompt_long_story = "サウナ好きの主人公が、何らかの理由でファンタジー風の異世界へ行き、そこで新たな人生をスタートさせる物語を書いてください。"

print("ストリーミング応答:")
for chunk in client.models.generate_content_stream(
    model=MODEL_ID,
    contents=prompt_long_story
):
    if chunk.text:  # チャンクにテキストコンテンツがあるかどうかを確認
        print(chunk.text, end="", flush=True)
print("\n")  # 最後に改行を追加
```

## 5. チャット（マルチターン会話）

chat クラスは、会話履歴を追跡するためのインターフェースを提供します。内部的には、`generate_content` メソッドを使用しています。つまり、chat_session.send_message() を呼び出すたびに、内部では「これまでの全会話」を毎回 generate_content に送り直すという、作業が自動的に行われています。


```python
chat_session = client.chats.create(model=MODEL_ID)

user_message1 = "ヨーロッパ旅行を計画しています。おすすめはありますか？"
print(f"ユーザー: {user_message1}")
response1 = chat_session.send_message(message=user_message1)
print(f"モデル: {response1.text}\n")
```


```python
user_message2 = "ニョッキが好きです。あまり高価じゃないものがいいです。"
print(f"ユーザー: {user_message2}")
response2 = chat_session.send_message(message=user_message2)
print(f"モデル: {response2.text}\n")
```


```python
user_message3 = "サウナはありますか？"
print(f"ユーザー: {user_message3}")
```


```python
# TODO: 
# さらに会話を続けましょう
# 
```


```python
# 会話履歴を表示
history = chat_session.get_history()
print(f"会話の合計メッセージ数: {len(history)}")
```

## 6. システム インストラクション

システム インストラクションを使用すると、モデルの動作と個性を定義できます。これらは会話全体で一貫して適用されます。

**システム インストラクションのベスト プラクティス:**
- 具体的に、明確に
- 役割とトーンを定義する
- フォーマットの好みを指定する
- 行動ガイドラインを設定する


```python
# システム インストラクションを使って、ロールやフォーマットを指定してみましょう。

system_instruction_marketing_pro = """
# 役割
あなたは、全く新しいフィットネスブランドを立ち上げる、マーケティングのプロです。

# ガイドライン
- 常に 3 つの異なる選択肢を提案してください。
- 抽象的な言葉を避け、ターゲット顧客がイメージしやすい具体的な言葉を選んでください。
- なぜその名前を提案するのか、論理的な根拠を必ず添えてください。

# フォーマット
提案は、以下の Markdown 形式を厳守してください。

## 提案1: [ジムの名前]
- **コンセプト:** (ジムのコンセプトを 1 〜 2 文で説明)
- **ターゲット層:** (どのような顧客を対象とするか)
- **ネーミングの由来:** (なぜこの名前にしたのか)
"""

# システム インストラクションを使ってコンテンツを生成
response_pro = client.models.generate_content(
    model=MODEL_ID,
    contents="全く新しいフィットネスジムとその名前を考えて",
    config=types.GenerateContentConfig(
        system_instruction=system_instruction_marketing_pro
    )
)

print(response_pro.text)
```

## 7. Gemini モデルのパラメータ

Gemini モデルのパラメータを使用して、生成の動作をカスタマイズします。これらを理解することで、応答を微調整することができます。

**パラメータガイド:**
- **Temperature (0.0-2.0)**: 回答の創造性やユニークさを調整します。例：低いと「犬」、高いと「ワンワン星人」のような答え。事実に基づいたコンテンツには 0.2 〜 0.4、創造的なコンテンツには 0.7 〜 1.0 を使用します
- **Top-p (0.0-1.0)**: 単語候補を確率の合計値で絞り込みます。例：0.9 なら多様な表現、0.1なら定番の言葉を選びます。
- **Top-k**: 単語候補を確率上位の個数で絞り込みます。例：30 なら候補多数、3 なら鉄板の 3 択の中から選びます。
- **Max output tokens**: 生成される文章の最大長（文字数）です。例：2000 なら長文、10 なら一言だけの返信になります。



```python
# Gemini モデルのパラメータを使用して、生成の動作をカスタマイズしましょう.

generation_config_dict = {
    "temperature": 0.5,         #回答の創造性やユニークさを調整します。
    "max_output_tokens": 2000,  # 生成される文章の最大長（文字数）です。
    "top_p": 0.5,               # 単語候補を確率の合計値で絞り込みます。
    "top_k": 20,                # 単語候補を確率上位の個数で絞り込みます。
}

response_config = client.models.generate_content(
    model=MODEL_ID,
    contents="フィットネスジムの名前を考えて",
    config=generation_config_dict
)

print(response_config.text)
```


```python
# TODO:
# Gemini モデルのパラメータを変えて、生成の動作をカスタマイズしてみましょう。
# 
```

## 8. 強大なコンテキストとファイルのアップロード

Gemini 2.5 には、100 万トークンという巨大なコンテキス トウィンドウを持っています。これがどれくらいの情報量かというと、具体的には下記のような例に相当します。

- コード: 50,000 行のコード
- 過去 5 年間に送信したすべてのテキストメッセージ
- 平均的な長さの英語の小説 8 冊
- 約 1 時間の動画データ

Google Colab の File API を使用すると、ファイルを Gemini API にアップロードし、リクエストのコンテキストとして使用できます。


```python
import requests

# サンプルテキストファイルをダウンロード
sample_text_url = "https://www.gutenberg.org/files/74/74-0.txt"  # トム・ソーヤーの冒険
response_req = requests.get(sample_text_url)

# ファイルに保存
with open("sample_book.txt", "w", encoding="utf-8") as f:
    f.write(response_req.text)

# ファイルを Gemini API にアップロード
try:
    myfile = client.files.upload(file="sample_book.txt")
    print(f"ファイルが正常にアップロードされました: {myfile.name}")
    
    # アップロードされたファイルをコンテキストとして使用してコンテンツを生成
    response = client.models.generate_content(
        model=MODEL_ID, 
        contents=[myfile, "この本の要点を 3 つ、箇条書きで簡潔にまとめてください。"])
    
    print("要約:")
    print(response.text)
    
    # 大規模なコンテキストのトークン使用量を確認
    print(f"\nトークン使用量: {response.usage_metadata.total_token_count}")
    
except Exception as e:
    print(f"ファイルのアップロード中にエラーが発生しました: {e}")
    print("ファイルが存在し、アクセス可能であることを確認してください")
```

## 9. 「本」 とチャット

「不思議の国のアリス」の本と「話す」ことができるインタラクティブなチャットを作成しましょう。ペルソナを設定し、本のテキストを会話のコンテキストとして使用します。

- 「不思議の国のアリス」のテキストをダウンロードします。
- 本のテキストファイル（`alice_in_wonderland.txt`）を `client.files.upload()` を使用して Gemini API にアップロードします。
- `client.chats.create()` を使用してチャットセッションを作成します。
- `chat.send_message()` を使用してチャットセッションに最初のメッセージを送信します。
- チャットセッションに少なくとも 1 つ質問を送信します。


```python
# 「不思議の国のアリス」の本とチャットするための準備
import requests

# 不思議の国のアリスをダウンロード
book_text_url = "https://www.gutenberg.org/files/11/11-0.txt"
try:
    response_book_req = requests.get(book_text_url)
    response_book_req.raise_for_status()  
    with open("alice_in_wonderland.txt", "w", encoding="utf-8") as f:
        f.write(response_book_req.text)
    print("本が正常にダウンロードされました！")
    
except requests.RequestException as e:
    print(f"本のダウンロード中にエラーが発生しました: {e}")
```


```python
# Gemini のモデルパラメータなどを設定して、チャットセッションを作成
chat = client.chats.create(
    model=MODEL_ID,
    config=types.GenerateContentConfig(
        system_instruction="あなたは大阪に住むユーモアあふれる書評家です。",
        temperature=1.2,  # 少しユニークな回答
    )
)
# 保存したファイルをアップロード
myfile = client.files.upload(file="alice_in_wonderland.txt")
# プロンプトの作成
prompt = f"""この本の要点を 5 つ、箇条書きで簡潔にまとめてください。

要約:
"""

response = chat.send_message([prompt, myfile])
print(prompt)
print(response.text)
```


```python
response = chat.send_message("X での投稿を作成してください。")
print(response.text)
```


```python
# TODO: 
# これまでのコードを参考にして、さまざまなテキストを生成してみましょう。
# 
```
