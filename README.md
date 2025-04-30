## AI Eyes Detection Project - Execution Plan & Log

### 🗓️ Daily Task Breakdown

| Day      | Task                | Tools                      | Description                                             | Status | Notes |
| -------- | ------------------- | -------------------------- | ------------------------------------------------------- | ------ | ----- |
| Day 1    | Environment setup   | conda, RETFound, AutoMorph | Prepare environments, install deps, set up data folders | ⬜     |       |
| Day 2    | Fine-tune RETFound  | torchrun                   | Train on labeled data                                   | ⬜     |       |
| Day 3 AM | Run AutoMorph       | AutoMorph                  | Extract vessel features                                 | ⬜     |       |
| Day 3 PM | Combine features    | pandas, sklearn            | Merge RETFound + AutoMorph for ML                       | ⬜     |       |
| Day 4 AM | Generate heatmaps   | Transformer-Explainability | Visualize ViT attention                                 | ⬜     |       |
| Day 4 PM | Evaluation & Report | sklearn, matplotlib        | Metrics, analysis, and summary                          | ⬜     |       |

### 🔗 Repos

-   [RETFound GitHub](https://github.com/rmaphoh/RETFound_MAE)
-   [AutoMorph GitHub](https://github.com/rmaphoh/AutoMorph)
-   [Transformer-Explainability GitHub](https://github.com/hila-chefer/Transformer-Explainability)
