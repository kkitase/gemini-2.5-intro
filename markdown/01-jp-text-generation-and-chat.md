
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/01-text-generation-and-chat.ipynb)

# パート1 - テキスト生成とチャット

このパートでは、`google-genai` SDK を使用した Gemini API によるテキスト生成に焦点を当て、基本的なプロンプト、チャットの対話、ストリーミング、設定について説明します。

[セットアップと認証](solution_00_setup_and_authentication.md) のセクションを完了していることを確認してください。

```python
from google import genai
from google.genai import types
import os
import sys
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
else:
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY',None)

# APIキーでクライアントを作成
MODEL_ID = "gemini-2.5-flash-preview-05-20"
client = genai.Client(api_key=GEMINI_API_KEY)
```

## 1. 最初のプロンプトを送信する

```python
prompt = "サステナビリティを重視した新しいコーヒーショップの名前を3つ作成してください。"

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)

print("Geminiからの応答:")
print(response.text)
```

#### !! 演習: さまざまなプロンプトを送信する !!

さまざまな種類のプロンプトをGeminiモデルに送信し、その応答を観察する練習をします。利用可能な場合は、異なるモデルバージョンを試すこともできます。

タスク:
- ロボットについての短い詩を生成するようにGeminiに依頼するプロンプトを作成します。
- 「機械学習」を簡単な言葉で説明するようにGeminiに依頼するプロンプトを作成します。
- 他のモデル（例：`gemini-2.0-flash`）を試し、プロンプトを送信して結果を比較します。

```python
# TODO:
```

## 2. トークンの理解とカウント

トークンは、Geminiモデルがテキストを処理するために使用する基本的な単位です。トークンの使用状況を理解することは、次の点で重要です。
- **コスト管理**: 請求はトークン消費量に基づいています
- **コンテキスト制限**: モデルには最大トークン制限があります（例：Gemini 2.5 Proの場合は100万トークン）
- **パフォーマンスの最適化**: 入力が小さいほど、一般的に処理が高速になります

Geminiモデルの場合、1トークンは約4文字に相当し、100トークンは約60〜80の英単語に相当します。

### 生成前のトークンをカウントする

モデルに送信する前に入力のトークンをカウントして、コストを見積もり、制限内に収まるようにすることができます。

```python
prompt = "The quick brown fox jumps over the lazy dog."

# 入力のトークンをカウントする
# TODO: client.models.count_tokens() メソッドを呼び出します。
# MODEL_IDとプロンプトを渡すようにしてください。
# token_count = client.models.count_tokens(
#     model=...,
#     contents=...
# )
print(f"入力トークン: {token_count.total_tokens}")

# コストの見積もり（価格例 - 現在のレートを確認してください）
estimated_cost = token_count.total_tokens * 0.15 / 1_000_000
print(f"推定入力コスト: ${estimated_cost:.6f}")
```

### 生成後のトークンをカウントする

コンテンツを生成した後、詳細なトークン使用状況情報にアクセスできます。

```python
prompt = "人工知能についての俳句を詠んでください。"

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)

print(f"生成された俳句:\n{response.text}\n")

# トークン使用メタデータにアクセスする
usage = response.usage_metadata
print(f"入力トークン: {usage.prompt_token_count}")
print(f"思考トークン: {usage.thoughts_token_count}")
print(f"出力トークン: {usage.candidates_token_count}")

# 合計推定コストを計算する
total_cost = (usage.prompt_token_count * 0.15 + (usage.candidates_token_count + usage.thoughts_token_count) * 3.5) / 1_000_000
print(f"合計推定コスト: ${total_cost:.6f}")
```

## 3. `contents` によるテキスト理解

テキストを生成する最も簡単な方法は、モデルにテキストのみのプロンプトを提供することです。 `contents` は、単一のプロンプト、プロンプトのリスト、またはマルチモーダル入力の組み合わせにすることができます。

```python
response_capital = client.models.generate_content(
    model=MODEL_ID,
    contents="フランスの首都はどこですか？"
)
print(f"Q: フランスの首都はどこですか？\nA: {response_capital.text}")
```

```python
# TODO: client.models.generate_content() メソッドを呼び出します。
# contentsには、文字列のリストを指定します。
# 1. "ビーガンレストランの名前を3つ作成してください"
# 2. "都市: ベルリン"
# response_restaurant_berlin = client.models.generate_content(
#     model=MODEL_ID,
#     contents=[...]
# )
print(f"\nベルリンのビーガンレストラン名:\n{response_restaurant_berlin.text}")
```

## 4. ストリーミング応答

ストリーミングを使用すると、生成される応答を増分的に受信できるため、長い応答やチャットボットなどのリアルタイムアプリケーションで、より優れたユーザーエクスペリエンスを提供できます。

**ストリーミングを使用する場合:**
- インタラクティブなアプリケーション（チャットボット、アシスタント）
- 長いコンテンツの生成
- リアルタイムのユーザーフィードバック
- 体感パフォーマンスの向上

```python
prompt_long_story = "勇敢な騎士とフレンドリーなドラゴンについての短編小説を書いてください。"

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

SDKチャットクラスは、会話履歴を追跡するためのインターフェースを提供します。内部的には、同じ `generate_content` メソッドを使用しています。

```python
chat_session = client.chats.create(model=MODEL_ID)

user_message1 = "週末旅行を計画しています。ヨーロッパでのシティブレイクにおすすめはありますか？"
print(f"ユーザー: {user_message1}")
response1 = chat_session.send_message(message=user_message1)
print(f"モデル: {response1.text}\n")
```

```python
user_message2 = "歴史とおいしい食べ物が好きです。あまり高価でないものがいいです。"
print(f"ユーザー: {user_message2}")
# TODO: user_message2 を使用して chat_session.send_message() メソッドを呼び出します。
# response2 = chat_session.send_message(message=...)
```

```python
# 会話履歴を表示
history = chat_session.get_history()
print(f"会話の合計メッセージ数: {len(history)}")
```

## 6. システムインストラクション

システムインストラクションを使用すると、モデルの動作と個性を定義できます。これらは会話全体で一貫して適用されます。

**システムインストラクションのベストプラクティス:**
- 具体的に、明確に
- 役割とトーンを定義する
- フォーマットの好みを指定する
- 行動ガイドラインを設定する

```python
system_instruction_poet = "あなたは17世紀の有名な詩人で、ソネットを専門としています。弱強五歩格で応答し、雄弁で時代に合った言葉遣いをしてください。"

response_poet = client.models.generate_content(
    model=MODEL_ID,
    contents="現代のテクノロジーについてどう思いますか？",
    config=types.GenerateContentConfig(
        system_instruction=system_instruction_poet
    )
)
print(f"\n詩人モデルによる現代技術に関する応答:\n{response_poet.text}")
```

## 7. 生成設定

設定パラメータを使用して、生成の動作をカスタマイズします。これらを理解することは、特定のユースケースに合わせて応答を微調整するのに役立ちます。

```python
# 辞書を使用した設定
generation_config_dict = {
    "temperature": 0.2,      # 低いほど決定的、高いほど創造的
    "max_output_tokens": 2000, # 応答の長さを制限
    "top_p": 0.8,            # Nucleusサンプリング - トークン選択の多様性
    "top_k": 30,             # 最も可能性の高い上位30トークンを考慮

}

# TODO: client.models.generate_content() を呼び出す
# MODEL_ID、"環境に優しいスニーカーの新ブランドの非常に短いタグラインを作成してください" というプロンプト、
# および generation_config_dict を渡します。
# response_config = client.models.generate_content(
#     model=...,
#     contents=...,
#     config=...
# )
```

**パラメータガイド:**
- **Temperature (0.0-2.0)**: ランダム性を制御します。事実に基づいたコンテンツには0.2〜0.4、創造的なコンテンツには0.7〜1.0を使用します
- **Top-p (0.0-1.0)**: 多様性を制御します。値が低いほど焦点が絞られ、高いほど多様になります
- **Top-k**: トークンの選択肢を制限します。値が低いほど焦点が絞られ、高いほど多様になります
- **Max output tokens**: 過度に長い応答を防ぎ、コストを管理します

## 8. 長いコンテキストとファイルのアップロード

Gemini 2.5 Proには、100万トークンのコンテキストウィンドウがあります。実際には、100万トークンは次のようになります。

- 50,000行のコード（1行あたり標準80文字）
- 過去5年間に送信したすべてのテキストメッセージ
- 平均的な長さの英語の小説8冊
- 1時間のビデオデータ

File APIを使用すると、ファイルをGemini APIにアップロードし、リクエストのコンテキストとして使用できます。

```python
# テキストファイルの例（音声の例よりも信頼性が高い）
import requests

# サンプルテキストファイルをダウンロード
sample_text_url = "https://www.gutenberg.org/files/74/74-0.txt"  # トム・ソーヤーの冒険
response_req = requests.get(sample_text_url)

# ローカルファイルに保存
with open("sample_book.txt", "w", encoding="utf-8") as f:
    f.write(response_req.text)

# ファイルをGemini APIにアップロード
try:
    myfile = client.files.upload(file="sample_book.txt")
    print(f"ファイルが正常にアップロードされました: {myfile.name}")
    
    # アップロードされたファイルをコンテキストとして使用してコンテンツを生成
    response = client.models.generate_content(
        model=MODEL_ID, 
        contents=[myfile, "この本を3つの重要なポイントで要約してください"])
    
    print("要約:")
    print(response.text)
    
    # 大規模なコンテキストのトークン使用量を確認
    print(f"\nトークン使用量: {response.usage_metadata.total_token_count}")
    
except Exception as e:
    print(f"ファイルのアップロード中にエラーが発生しました: {e}")
    print("ファイルが存在し、アクセス可能であることを確認してください")
```

## 9. !! 演習: 「本」とチャットする !!

「不思議の国のアリス」という本と「話す」ことができるインタラクティブなチャットセッションを作成します。AIに特定のペルソナを設定し、本のテキストを会話のコンテキストとして使用します。

タスク: 
- 「不思議の国のアリス」のテキストをダウンロードします（ヘルパーコードブロックが提供されています）。
- 本のテキストファイル（`alice_in_wonderland.txt`）を `client.files.upload()` を使用してGemini APIにアップロードします。
- `client.chats.create()` を使用してチャットセッションを作成します。
- `chat.send_message()` を使用してチャットセッションに最初のメッセージを送信します。
- チャットセッションに少なくとも1つのフォローアップの質問（例：「音声配信のさまざまな方法を詳しく説明してください」）を送信し、その応答を印刷します。

```python
import requests

# 不思議の国のアリスをダウンロード
book_text_url = "https://www.gutenberg.org/files/11/11-0.txt"
try:
    response_book_req = requests.get(book_text_url)
    response_book_req.raise_for_status()  # 不正なステータスコードに対して例外を発生させる
    
    with open("alice_in_wonderland.txt", "w", encoding="utf-8") as f:
        f.write(response_book_req.text)
    print("本が正常にダウンロードされました！")
    
except requests.RequestException as e:
    print(f"本のダウンロード中にエラーが発生しました: {e}")
```

```python
# TODO:
```

```python
# TODO:
```

## まとめと次のステップ

**学習したこと:**
- 単一プロンプトに対する `client.models.generate_content()` を使用した基本的なテキスト生成
- より良いリソース管理のためのトークンカウントとコスト見積もり
- ユーザーエクスペリエnciaを向上させるための `generate_content_stream()` を使用したストリーミング応答
- `client.chats.create()` とチャットセッションを使用したマルチターン会話
- 一貫したモデルの動作と個性のためのシステムインストラクション
- 応答を微調整するための生成設定パラメータ
- File APIを使用した長いコンテキストの処理とファイルのアップロード
- 本番アプリケーションのエラー処理とベストプラクティス

**重要なポイント:**
- トークンの使用状況を監視してコストを管理し、制限内に収める
- インタラクティブなアプリケーションと長い応答にはストリーミングを使用する
- ユースケース（事実に基づいたコンテンツか創造的なコンテンツか）に基づいてパラメータを設定する
- 堅牢なアプリケーションのために適切なエラー処理を実装する
- システムインストラクションは、動作とトーンを設定するのに強力です

**次のステップ:** [パート2：マルチモーダル機能](https://github.com/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/02-multimodal-capabilities.ipynb) に進みます [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/02-multimodal-capabilities.ipynb)

**その他のリソース:**
- [テキスト生成ガイド](https://ai.google.dev/gemini-api/docs/text-generation)
- [トークンカウントガイド](https://ai.google.dev/gemini-api/docs/tokens)
- [長いコンテキストのドキュメント](https://ai.google.dev/gemini-api/docs/long-context)
- [File API ドキュメント](https://ai.google.dev/gemini-api/docs/files)