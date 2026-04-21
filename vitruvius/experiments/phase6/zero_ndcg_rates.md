# Zero-nDCG@10 rates (per encoder x dataset)

Values are the fraction of evaluation queries for which the
encoder's top-10 list contains no qrels-relevant document.

| encoder            | nfcorpus   | scifact   | fiqa   |
|:-------------------|:-----------|:----------|:-------|
| minilm-l6-v2       | 31.0%      | 20.7%     | 34.3%  |
| bert-base          | 32.8%      | 26.7%     | 40.9%  |
| gte-small          | 29.7%      | 14.0%     | 33.3%  |
| lstm-retriever     | 50.2%      | 52.0%     | 78.7%  |
| conv-retriever     | 58.5%      | 73.7%     | 90.4%  |
| mamba-retriever-fs | 47.7%      | 49.0%     | 77.3%  |

Raw counts (failures / total):

| encoder            | nfcorpus   | scifact   | fiqa    |
|:-------------------|:-----------|:----------|:--------|
| minilm-l6-v2       | 100/323    | 62/300    | 222/648 |
| bert-base          | 106/323    | 80/300    | 265/648 |
| gte-small          | 96/323     | 42/300    | 216/648 |
| lstm-retriever     | 162/323    | 156/300   | 510/648 |
| conv-retriever     | 189/323    | 221/300   | 586/648 |
| mamba-retriever-fs | 154/323    | 147/300   | 501/648 |
