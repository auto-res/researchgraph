# LLM Componentの開発方法
ここではLLM Componentの実装方法について説明します

## 導入
- すべてのLLM Componentはllm_component.pyのLLMComponentクラスを用いて作成されます．
- LLMComponentクラスだけで作成できないComponentについては処理を分解するか，別のComponentとして実装してください．
- LLMの実行回数は一回でなくても問題ありません．

## 実装
- LLMComponentの実装で必要なものは所定の形式を守ったJSONファイルのみ．
- JSONファイルの形式
    - 1回LLMを実行する場合
        - JSONファイルの例
        ```python
        {
            "input": ["source", "language"],
            "output": ["translated_source"],
            "prompt": "<source>{source}</source>\n<language>{language}</language>\n\n\nsourceタグで与えられた文章をlanguageで指定された言語に翻訳してtranslated_sourceタグを用いて出力せよ."
        }
        ```
        - inputおよび，outputとして与える変数を設定し，リスト形式で記載する
        - プロンプトにはinputとして与えられている変数を埋め込む箇所に同じ変数名で{}を使い記載する．sourceというinputがある場合必ずプロンプトに{source}を記載する．また同じ変数名のタグを使い\<source>{source}\</source>のように変数の箇所を挟む．
        - 出力形式は，outputの変数名と同じタグで出力を囲むように指示しる．上の例では「translated_sourceタグを用いて出力せよ」がその指示にあたる．
    - 2回以上LLMを実行する場合
        - Jsonファイルの例
        ```python
        {
        "input": [
            ["source", "language"],
            ["translated_source2"]
        ],
        "output": [
            ["translated_source1", "translated_source2"],
            ["translated_source3"]
        ],
        "prompt": [
            "<source>{source}</source>\n<language>{language}</language>\n\n\nsource_text タグで与えられた文章をlanguageで指定された言語に翻訳してtranslated_source1タグを用いて出力せよ．さらに，翻訳したものをドイツ語に翻訳しtranslated_source2タグを用いて出力してください．",
            "<translated_source2>{translated_source2}\<translated_source2>\n\n\ntranslated_source2タグで与えられた文章をフランス語に翻訳してtranslated_source3タグを用いて出力せよ．"
        ]
        }
        ```
        - 基本的な設定は「1回LLMを実行する場合」と同じ
        - 実行する回数分の要素を持つリストとして定義する．3回以上LLMを実行する場合でも同様
