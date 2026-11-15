# C — Independent Evaluation and Results Analysis

**Project:** When Does Synthetic Data Help? A Generation–Training–Evaluation Toolkit for 20-Shot Image Classification

**Protocol:** 0.4-frozen

**Run ID:** synthetic_v2

**Dataset:** CIFAR-10

**Evaluation:** 10,000 official real test images

**Repetitions:** Three paired seeds (0, 1, 2)

## 1. Evaluation Methodology

The experiment evaluates whether synthetic image augmentation improves classification under a limited real-data budget.

For each random seed, 20 real training images per CIFAR-10 class were selected. Five experimental conditions shared the same real training subset and paired training setup:

- Real Only: no synthetic images.
- Random 50: 50 randomly selected synthetic images per class.
- CLIP 50: 50 CLIP-ranked synthetic images per class.
- Random 150: 150 randomly selected synthetic images per class.
- CLIP 150: 150 CLIP-ranked synthetic images per class.

All 15 classifiers used a frozen ImageNet-pretrained ResNet-18 feature extractor and trained only the linear classification head.

Final evaluation used the official CIFAR-10 test set, which was excluded from model training, checkpoint selection and hyperparameter tuning.

Top-1 accuracy was independently recomputed from the released prediction records:

**Accuracy = Number of correct predictions / Number of test images**

For each condition, we report the mean and sample standard deviation across three seeds. Improvements over the Real Only baseline are calculated using matched seeds:

**Paired Gain (pp) = 100 × (Accuracy_condition − Accuracy_real_only)**

Sample standard deviations use `ddof=1`. These standard deviations are not confidence intervals.

## 2. Overall Classification Results

| Condition | Mean Accuracy (%) | Sample SD (pp) | Paired Gain (pp) |
|---|---:|---:|---:|
| Real Only | 71.44 | 0.314 | — |
| Random 50 | 75.08 | 0.737 | +3.640 |
| CLIP 50 | **75.28** | 0.763 | **+3.840** |
| Random 150 | 75.06 | 0.584 | +3.617 |
| CLIP 150 | 75.26 | 0.577 | +3.820 |

All four synthetic-data conditions achieved higher mean test accuracy than the Real Only baseline.

The improvements ranged from approximately 3.62 to 3.84 percentage points. All three paired seeds showed positive improvements in every synthetic-data condition.

Among the five trained-classifier conditions, CLIP 50 achieved the highest mean accuracy, at 75.28%.

These findings suggest that synthetic augmentation can improve classification performance in the tested 20-shot setting. However, the experiment does not establish that the same improvement will generalize to other datasets, models or synthetic generation configurations.

## 3. Selection Method and Synthetic Data Quantity

To distinguish the effects of selection strategy and retained sample count, four direct paired comparisons were calculated.

| Paired Comparison | Mean Difference (pp) | Sample SD (pp) |
|---|---:|---:|
| CLIP 50 − Random 50 | +0.200 | 0.221 |
| CLIP 150 − Random 150 | +0.203 | 0.123 |
| Random 150 − Random 50 | −0.023 | 0.153 |
| CLIP 150 − CLIP 50 | −0.020 | 0.190 |

### 3.1 Does CLIP selection outperform random selection?

At equal retained sample counts, CLIP selection achieved slightly higher mean accuracy than random selection.

The average advantage was approximately 0.20 percentage points for both the 50-image and 150-image conditions.

However, these differences were small relative to the limited number of repeated experiments. At 50 images per class, CLIP performed slightly worse than Random for seed 0.

Therefore, the evidence supports only a small descriptive advantage for CLIP ranking in this experiment. It does not establish a statistically reliable or broadly generalizable superiority.

### 3.2 Does retaining more synthetic data improve performance?

Increasing the number of retained synthetic images from 50 to 150 per class produced almost no change in mean accuracy.

The mean differences were −0.023 percentage points for Random selection and −0.020 percentage points for CLIP selection.

Thus, the additional retained images did not provide an observable average benefit under the fixed source-balanced training objective and equal update budget.

These results suggest diminishing returns from larger retained synthetic subsets within the tested range. They do not prove that additional synthetic data is generally ineffective.

## 4. Per-Class Performance

Synthetic augmentation produced different effects across CIFAR-10 classes.

For the CLIP 50 condition, the largest mean improvements over the matched Real Only baseline included:

| Class | Mean Accuracy (%) | Mean Gain (pp) |
|---|---:|---:|
| Deer | 73.40 | +12.47 |
| Truck | 89.83 | +5.90 |
| Dog | 73.67 | +5.50 |
| Horse | 74.23 | +4.20 |
| Frog | 86.00 | +4.13 |
| Automobile | 85.53 | −1.00 |
| Ship | 78.83 | −1.97 |

Deer benefited particularly strongly, with a mean improvement exceeding 12 percentage points.

In contrast, Automobile and Ship showed negative mean changes in the CLIP 50 condition. This indicates that synthetic augmentation does not improve every category uniformly.

The per-class results also show that CLIP selection does not consistently outperform random selection for every category.

These differences may motivate further investigation into synthetic image quality, class ambiguity and feature-domain mismatch. However, the observed accuracy differences alone do not establish these mechanisms as causal explanations.

## 5. CLIP Zero-Shot Reference

The CLIP zero-shot classifier achieved **88.21% Top-1 accuracy** on the same official CIFAR-10 test set.

This exceeds the highest mean accuracy of the trained linear classifiers, which was 75.28% for CLIP 50.

Nevertheless, these results are not directly equivalent comparisons of data augmentation techniques. CLIP zero-shot classification uses a different pretrained model and a different inference procedure.

The result demonstrates that a strong zero-shot alternative should be considered when evaluating the practical usefulness of the proposed toolkit.

The toolkit therefore provides evidence of improvement over its matched ResNet-18 baseline, but cannot claim to be the best classification approach for this dataset.

## 6. Computational Cost Analysis

Recorded G3 production costs and G4 final evaluation time were independently reconciled with their respective execution records.

| Stage | Recorded Time (seconds) | Time (minutes) |
|---|---:|---:|
| SD1.5 candidate generation | 2551.31 | 42.52 |
| CLIP selection | 39.97 | 0.67 |
| ResNet-18 feature preparation | 69.53 | 1.16 |
| Fifteen linear-head training runs | 8.42 | 0.14 |
| G4 final evaluation | 219.36 | 3.66 |
| **Recorded stage total** | **2888.59** | **48.14** |

Synthetic image generation accounted for approximately **88.3%** of these recorded stage times.

This indicates that candidate generation was the dominant recorded computational expense. In contrast, training the linear classification heads required comparatively little additional time after feature preparation.

The 2,000 generated candidate images were shared across seeds and experimental conditions. Their generation cost must therefore not be counted independently for each classifier.

The generation time includes recorded failed invocations, loading, warm-up and recovery. The G4 evaluation time includes its internal operations and should not be added again to the separate CLI wall-clock duration of approximately 221.52 seconds.

The 48.14-minute total is a sum of selected recorded execution stages, **not the total development time or complete deployment cost**. It excludes environment setup, downloads, earlier smoke tests, software development, diagnostic work, manual reviews, packaging and other activities.

Because retaining 150 rather than 50 synthetic images per class produced no additional average accuracy benefit, the smaller retained subset is a reasonable choice within the tested configuration. This does not imply that generating the shared candidate pool itself becomes cheaper without modifying the generation protocol.

## 7. Independent Verification and Reproducibility

The C-side independent review was conducted locally using Windows 10 and Python 3.12.15.

The review included:

1. SHA-256 validation of the G3 training archive and G4 prediction archive.
2. Independent consistency checks of 2,000 candidate records, CLIP scores, saved rankings and nested selection sets.
3. Verification of 16 G4 prediction files containing 160,000 total prediction records.
4. Validation of test sample IDs, class balance, stored prediction hashes and cross-file label consistency.
5. Recalculation of per-run Top-1 accuracy and comparison with released evaluation summaries.
6. Recalculation of paired gains, sample standard deviations, direct contrasts and per-class accuracy.
7. Reconciliation of recorded computational costs.
8. Cross-checking frozen training configurations, real-data splits, paired initialization, update counts, source hashes and static training logic.

These checks passed within their defined scopes.

However, C did not independently rerun CLIP inference, generate new synthetic images, retrain the 15 classifiers or reproduce their official predictions through separate GPU inference.

The training-fairness inspection checked recorded artifacts and source consistency. It should not be described as an independent numerical reproduction of the full GPU training process.

The audit scripts were prepared with AI assistance and executed locally by C, with their results and execution logs retained.

### Related B-side numerical reproducibility finding

The separate B-side review reproduced all 15 linear-head optimizations with a maximum weight discrepancy of approximately 4.77 × 10⁻⁷, within its predefined 10⁻⁶ threshold.

However, CPU-extracted ResNet features differed from the sealed CUDA features by as much as approximately 2.01 × 10⁻⁴, exceeding the predefined absolute tolerance of 10⁻⁵.

This strict feature-reproduction check therefore did not pass.

The B review reported no changes among 3,000 training-reference predictions when using the re-extracted features. A subsequent limited GPU diagnostic supported a backend-precision explanation for the observed difference, without establishing complete cross-platform equivalence.

The frozen models, training inputs and formal G4 predictions were not modified by these diagnostics.

## 8. Limitations and Interpretation

Several limitations should be considered when interpreting the findings.

**Limited experimental scope.** The study uses one dataset, one real-data budget, one synthetic generator and one frozen feature backbone. Results should not be generalized to other settings without further experiments.

**Only three paired seeds.** The reported sample standard deviations describe variation across three repeated splits and training runs. They should not be interpreted as confidence intervals or evidence of statistical significance.

**Shared synthetic pool.** All seeds reuse the same synthetic candidate pool and rankings. The experiments therefore do not measure variability caused by generating different synthetic pools.

**Pretrained model knowledge.** ResNet-18 and CLIP rely on external pretraining. This is not classification learned from only 200 labelled images without prior information.

**Generated-label and domain uncertainty.** Class-conditioned generation and CLIP selection do not guarantee that every generated image has the intended class or matches the distribution of official CIFAR-10 test images.

**Incomplete human evaluation.** The planned two-annotator blinded image-label audit and non-author usability trial have not been confirmed as completed in the inspected records. Their results must not be claimed without corresponding evidence.

**Cross-platform numerical differences.** The B-side feature-reproduction discrepancy remains a documented limitation, even though the checked training-reference predictions were unchanged.

## 9. Conclusion

Under the tested 20-shot CIFAR-10 configuration, synthetic augmentation improved the mean accuracy of the frozen-feature classifiers by approximately 3.6–3.8 percentage points compared with the matched Real Only baseline.

CLIP-based selection achieved a small mean advantage of approximately 0.2 percentage points over random selection at equal retained counts, while increasing the number of synthetic images from 50 to 150 per class yielded no additional average improvement.

The dominant recorded computational cost was synthetic image generation. Moreover, the 88.21% CLIP zero-shot reference demonstrated that alternative pretrained approaches can outperform the trained classifiers in this setting.

Overall, the toolkit provides a reproducible way to evaluate the downstream value and computational cost of synthetic augmentation. Its results support conditional, task-specific use rather than a general claim that more generated images or CLIP ranking will always improve classification.

## 10. Supporting Evidence

**Independent C-side analysis outputs:**

- `07_g4_summary.csv` — Overall accuracy and paired gains.
- `08_g4_per_class.csv` — Per-class results.
- `10_g4_accuracy.png` — Five-condition accuracy comparison.
- `11_g4_paired_gain.png` — Paired improvements over baseline.
- `12_g4_cost_summary.csv` — Recorded computational costs.
- `13_g4_direct_comparisons.csv` — Direct paired contrasts.
- `13_g4_direct_comparisons.png` — Selection-method and quantity comparisons.
- `14_g4_per_class_gain.png` — Per-class gains across four synthetic conditions.
- `15_training_fairness_audit.py` — Training fairness and frozen-source audit.

**Primary project records:**

- `configs/study.json`
- `shared/synthetic_v2/stage_costs.json`
- G4 Release: `g4-official-evaluation-v0.4`
- `report/B_v04_review.md`
- `shared/b_v04_review/`

Original local execution logs are retained separately. They contain local filesystem information and are not included in the public repository.

The project configuration, sealed model references and formal evaluation records remain unchanged.
