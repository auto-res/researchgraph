translater1_setting = {
    "input": ["source", "language"],
    "output": ["translation1"],
    "prompt": """
    <source>
    {source}
    </source>
    <language>
    {language}
    </language>
    <rule>
    sourceタグで与えられた文章を languageで指定された言語に翻訳して translation1タグを用いて出力せよ．
    </rule>
""",
}

translater2_setting = {
    "input": ["source"],
    "output": ["translation2_1", "translation2_2"],
    "prompt": """
    <source>
    {source}
    </source>
    <rule>
    sourceタグで与えられた文章をフランス言語に翻訳してtranslation2_1タグを用いて出力せよ．さらに，ドイツ語にも翻訳しtranslation2_2タグを用いて出力してください．
    </rule>
""",
}

translater3_setting = {
    "input": [["source"], ["translation3_1"]],
    "output": [["translation3_1", "translation3_2"], ["translation3_3"]],
    "prompt": [
        """
    <source>
    {source}
    </source>
    <rule>
    sourceタグで与えられた文章をスペイン言語に翻訳してtranslation3_1タグを用いて出力せよ．さらに，翻訳したものをポルトガル語に翻訳しtranslation3_2タグを用いて出力してください．
    </rule>
    """,
        """
    <translation3_1>
    {translation3_1}
    </translation3_1>
    <rule>
    translation3_1タグで与えられた文章をロシア語に翻訳してtranslation3_3タグを用いて出力せよ．
    </rule>
    """,
    ],
}
