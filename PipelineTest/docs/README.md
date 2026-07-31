# PipelineTest/docs

## Report & Documentation

LaTeX subsections for the research report on hallucination detection in discrete diffusion language models.

### Files

| File | Report Section | Content |
|------|---------------|---------|
| `report.md` | Full report | Complete assembled document |
| `subsection_baseline_features.md` | §2.1 | Mean, variance, dynamic features definitions |
| `subsection_semantic_clustering.md` | §2.2 | STC clustering, semantic entropy |
| `subsection_markovian_modelisation.md` | §2.3 | Markov model, entropy decay $S(t) \approx Cte^{-t/\tau}$, feature derivation |
| `subsection_ornstein_uhlenbeck.md` | §2.4 | OU process, AR(1) discretisation, estimation, features |

### Report Structure

```
§1. Experimental Setup
    - Dataset (TriviaQA)
    - Diffusion-based token generation
    - Information collected during sampling
    - Hallucination evaluation (TraceDet / Qwen judge)
    - Models and samples (LLaDA, Dream)

§2. Features Generation
    §2.1 Baseline Features (mean, variance, dynamic)
    §2.2 Semantic Token Clustering
    §2.3 Markovian Modelisation (theoretical motivation)
    §2.4 Ornstein-Uhlenbeck & AR(1) Discretisation
```
