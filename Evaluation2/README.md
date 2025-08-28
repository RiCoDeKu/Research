## 「No Silver Bullet-Prompt」実験 
##### ～単一の最適プロンプトは存在しないことの証明～
#### 目的: 
「プロンプトエンジニアリングだけでは限界がある」ことを定量的に示す

#### 手法:

現在使用している「高密度プロンプト (Prompt-Fine)」に加え、逆の指示をする「低密度プロンプト (Prompt-Coarse)」を新たに作成する。
##### 例:
 **Prompt-Coarse:** 
```
Analyze the entire video and extract only the main stages of the task. 
Combine fine-grained actions and summarize them into approximately 3 to 5 major steps.
```

 **Prompt-Fine:**
```
Analyze the video frame-by-frame to identify fine-grained human action events. For each action, provide a highly precise segmentation with minimal interval length, including both the exact start and end frame numbers, and a clear, detailed narration in a complete sentence.
    Detect even the smallest scene changes from frame 0 to frame {str(total_frame)} and perform as many scene divisions as possible.
    Note: This video was created at {fps}fps

    Please format your output as follows:
    [Format]
    start_frame;end_frame;narration
    [Example]
    0;120;prepare_dressing
    120;170;cut_cheese
    170;199;cut_and_mix_ingredients
```
これら2種類のプロンプトを、3つのデータセット（Epic-Kitchens, 50salads, ATA）すべてに適用し、F1スコアを計測する。

#### 期待される結果:


|データセット|Prompt-Fine (既存)|Prompt-Coarse (新規)|
|:----------|:-----------------|-------------------|
|Epic-Kitchens|高スコア (≥0.8)|低スコア (<0.6)|
|50salads|低スコア (≤0.6)|高スコア (≥0.7)|
|ATA|高スコア (≥0.8)|中〜高スコア (≥0.7)|

#### 結論:
この結果が得られれば、「データセットの特性に応じて最適なプロンプトは異なり、万能なプロンプトは存在しない」と結論付けられる。これは、「なぜこの研究が必要なのか？」という問いに対する、動かぬ証拠となる。

