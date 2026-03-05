# Data Upload Requirements for SuppQATrain

## Overview
This environment requires the following data to be uploaded to OpenReward cloud storage.

## Directory Structure
```
/orwd_data/
└── train.parquet
```

## File Descriptions
- **train.parquet**: Dataset containing scientific supplementary material QA pairs with columns: `id`, `question`, `answer`, `source_doi`, `key_passage`, `domain`, `supp_type`

## Upload Instructions
Upload `train.parquet` to the `GeneralReasoning/SuppQATrain` namespace on OpenReward.
