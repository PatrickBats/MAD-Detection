# Constant Proportion Synthetic Loop Experiment

## Overview

This experiment tests how different proportions of synthetic data affect Model Autophagy Disorder (MADness) in iterative generative model training. Unlike the accumulating approach, the total training set size stays **constant at 60,000 samples** across all generations.

## Experimental Design

### Key Principle: Fixed Proportion per Experiment

Each experiment run uses a **fixed synthetic proportion** that remains constant across all generations:

| Experiment | Real Samples | Synthetic Samples | Total | Description |
|------------|-------------|-------------------|-------|-------------|
| p=0.0      | 60,000      | 0                 | 60,000 | Baseline (no synthetic) |
| p=0.25     | 45,000      | 15,000            | 60,000 | 25% synthetic |
| p=0.5      | 30,000      | 30,000            | 60,000 | 50% synthetic (matches Paper-Style) |
| p=0.75     | 15,000      | 45,000            | 60,000 | 75% synthetic |
| p=0.9      | 6,000       | 54,000            | 60,000 | 90% synthetic |
| p=1.0      | 0           | 60,000            | 60,000 | Full synthetic loop |

### Generation Structure

For each experiment with proportion `p`:

```
Gen 0: Train on 100% REAL data (60k samples)
       → Generate 60k synthetic samples for next generation

Gen 1: Train on (1-p)*60k real + p*60k synthetic from Gen 0 = 60k total
       → Generate 60k synthetic samples for next generation

Gen 2: Train on (1-p)*60k real + p*60k synthetic from Gen 1 = 60k total
       → Generate 60k synthetic samples for next generation

...and so on
```

**Key**: Synthetic data comes ONLY from the previous generation (not accumulated).

## Why This Design Addresses Confounding

### The Concern
> "Wouldn't we not know why the change is happening - could it be from the change in data proportion OR the change in quality of the synthetic data?"

### The Answer

**Within a single experiment run:**
- The proportion is FIXED (e.g., always 50% synthetic)
- The only thing that changes across generations is the **quality** of the synthetic data
- If we see degradation, it's caused by the iterative training on degrading synthetic outputs

**Across different experiments:**
- We run separate experiments with different fixed proportions
- By comparing p=0.25 vs p=0.5 vs p=0.75, we can measure how proportion affects MADness
- The proportion is the **controlled independent variable**

### Separating the Variables

| Variable | How It's Controlled |
|----------|---------------------|
| Synthetic proportion | Fixed within each run; varied between runs |
| Training set size | Always 60k (constant) |
| Synthetic data quality | Allowed to vary naturally (this IS the phenomenon we study) |
| Real data | Same MNIST dataset throughout |

## Expected Results Based on Prior Work

| Proportion | Expected Behavior | Reasoning |
|------------|------------------|-----------|
| p=0.0 | Stable (baseline) | No synthetic data, no MADness possible |
| p=0.25 | Stable | Real data dominates, "grounds" quality |
| p=0.5 | Stable or slight degradation | Similar to Paper-Style (showed stability) |
| p=0.75 | Moderate degradation | Less real data to anchor quality |
| p=0.9 | Significant degradation | Almost no real data grounding |
| p=1.0 | Severe degradation | Same as FullSynthetic (ER: 3.63→1.50, FID: 5.73→36.05) |

## Running the Experiments

```bash
# Baseline - no synthetic
python main_constant_proportion.py --proportion 0.0 --generations 9

# Low synthetic
python main_constant_proportion.py --proportion 0.25 --generations 9

# Balanced (matches Paper-Style)
python main_constant_proportion.py --proportion 0.5 --generations 9

# High synthetic
python main_constant_proportion.py --proportion 0.75 --generations 9

# Very high synthetic
python main_constant_proportion.py --proportion 0.9 --generations 9

# Full synthetic (should replicate FullSynthetic results)
python main_constant_proportion.py --proportion 1.0 --generations 9
```

Each experiment saves to a separate directory: `./data/gan_outputs_p{proportion*100}/`

## Metrics to Track

1. **FID Score** per generation - measures distribution similarity to real MNIST
2. **Effective Rank** of Jacobian - measures latent space diversity/mode collapse
3. **Visual Quality** - sample grids showing low vs high ER samples

## Comparison with Other Approaches

| Approach | Training Set Size | Synthetic Source | Proportion |
|----------|------------------|------------------|------------|
| **Constant Proportion** | 60k (fixed) | Previous gen only | Fixed per experiment |
| Accumulating | 60k→480k (grows) | All previous gens | Varies: 0%→87.5% |
| Paper-Style | 120k (fixed) | Previous gen only | Fixed at 50% |
| FullSynthetic | 60k (fixed) | Previous gen only | Fixed at 100% (after Gen 0) |

## Key Insight

The **constant proportion design** isolates the effect of synthetic data proportion by:
1. Keeping total training size constant (controls for dataset size effects)
2. Using only previous-gen synthetic (controls for accumulation effects)
3. Running separate experiments per proportion (controls for proportion as independent variable)

This allows us to determine the **critical proportion threshold** at which MADness begins to manifest.
