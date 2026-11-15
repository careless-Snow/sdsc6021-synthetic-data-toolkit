# C — Independent G4 Evaluation: Reproduction and Handoff

## Scope and provenance

This directory contains **C's independent analysis outputs** for the SDSC6021 20-shot CIFAR-10 synthetic-augmentation study, protocol `0.4-frozen`, run ID `synthetic_v2`.

- Official G4 prediction archive: [G4 release `g4-official-evaluation-v0.4`](https://github.com/kanekikenny/sdsc6021-synthetic-data-toolkit/releases/tag/g4-official-evaluation-v0.4), `g4_predictions_and_run_records.zip` (16 prediction files; 160,000 test predictions).
- G4 official execution code: `26e107feba6c4c4c5f447f84cada707e140d738c` (recorded by the release); this analysis does **not** modify the frozen experiment.
- Recorded G3 costs: [`../stage_costs.json`](../stage_costs.json). The G4 report and invocation records are in the G4 release archive.
- Analysis environment actually used: **Windows 10, Python 3.12.15**, with **Matplotlib 3.11.2** for figure generation. `14_g4_per_class_gain_plot.py` also imports NumPy. A's sealed G4 evaluation used a different environment; no matching GPU environment is required for these post hoc checks.
- The original downloaded ZIP and unsanitized PowerShell transcripts are kept **locally**, not committed to this public repository.

The checks independently recalculate statistics **from stored predictions**; they do not retrain a model, rerun CLIP, reproduce GPU inference, or independently authenticate the CIFAR-10 labels against newly downloaded original images.

**A-side acceptance, 2026-11-16:** all nine submitted scripts were rerun against the SHA-verified official archive in an isolated review directory. All four CSV contents matched the submission, and all four regenerated figures matched the submitted pixels exactly. The [acceptance record](../audits/c_pr4_acceptance.json) preserves script/log hashes, review scope and dependency provenance. No formal input, classifier or prediction changed. This verifies the analysis layout and outputs; it is not a fresh installation/user-usability trial.

## Deliverables

### Tables

| File | Content |
| --- | --- |
| [`tables/07_g4_summary.csv`](tables/07_g4_summary.csv) | Mean test accuracy, sample SD, and same-seed gain relative to `real_only` for all five conditions |
| [`tables/08_g4_per_class.csv`](tables/08_g4_per_class.csv) | Ten-class accuracy and paired gains for each condition |
| [`tables/12_g4_cost_summary.csv`](tables/12_g4_cost_summary.csv) | Recorded G3 production-stage and G4 evaluation durations |
| [`tables/13_g4_direct_comparisons.csv`](tables/13_g4_direct_comparisons.csv) | Direct same-seed contrasts: CLIP versus Random and 150 versus 50 per class |

### Figures

| File | Content |
| --- | --- |
| [`figures/10_g4_accuracy.png`](figures/10_g4_accuracy.png) | Five-condition accuracy (mean ± sample SD across three seeds) |
| [`figures/11_g4_paired_gain.png`](figures/11_g4_paired_gain.png) | Same-seed gain over real-only, including individual seed values |
| [`figures/13_g4_direct_comparisons.png`](figures/13_g4_direct_comparisons.png) | Direct selection-method and synthetic-quantity contrasts |
| [`figures/14_g4_per_class_gain.png`](figures/14_g4_per_class_gain.png) | Per-class gains under four synthetic-data conditions |

The narrative write-up is in [`../../../report/C_results_evaluation.md`](../../../report/C_results_evaluation.md).

## How to rerun the independent checks (Windows PowerShell)

These scripts intentionally preserve the **working-directory layout used for the original C-side execution**. Run them from an independent folder named `G4_evaluation`, **not** from within `code/audits/`. They use paths relative to the current working directory.

1. Download a copy of this project's branch containing the C scripts (or its merged `main`) into `G4_evaluation/project_source/`; this copy must contain `code/`, `configs/`, `shared/`, etc., directly inside `project_source/`.
2. Download `g4_predictions_and_run_records.zip` from the G4 release into `G4_evaluation/`. Verify its SHA-256 against the release's `SHA256SUMS.txt` / `shared/synthetic_v2/g4/delivery.json` before proceeding. The published ZIP SHA-256 for this run is:

   `91045de3b946dccd21d9b6810f06309b1f86dc09ba4d814a158a4f9c078f6807`

3. From `G4_evaluation`, prepare the scripts and the cost input:

   ```powershell
   New-Item -ItemType Directory -Force .\C_evidence | Out-Null
   Copy-Item .\project_source\code\audits\*.py .\C_evidence\ -Force
   Copy-Item .\project_source\shared\synthetic_v2\stage_costs.json .\stage_costs.json -Force
   Get-FileHash .\g4_predictions_and_run_records.zip -Algorithm SHA256
   ```

4. Use a Python 3.12 environment. Python 3.12.15 was used in C's run; the standard-library-only audits do not require a GPU. The figure scripts require Matplotlib, and the per-class figure script requires NumPy. Run the following from `G4_evaluation`, replacing `python` with the path to your interpreter if necessary:

   ```powershell
   python .\C_evidence\06_g4_prediction_audit.py
   python .\C_evidence\07_g4_stats_audit.py
   python .\C_evidence\08_g4_per_class.py
   python .\C_evidence\10_g4_accuracy_plot.py
   python .\C_evidence\11_g4_paired_gain_plot.py
   python .\C_evidence\12_g4_cost_analysis.py
   python .\C_evidence\13_g4_direct_comparisons.py
   python .\C_evidence\14_g4_per_class_gain_plot.py
   python .\C_evidence\15_training_fairness_audit.py
   ```

   The `07` script creates the CSV required by `10` and `11`; the `08` script creates the CSV required by `14`. The `12` script needs `stage_costs.json`. The `15` script inspects the complete source tree in `project_source/` and its frozen shared records. Scripts produce their outputs under `C_evidence/`.

5. For independent evidence, capture the actual interpreter version and terminal output; e.g.:

   ```powershell
   Start-Transcript -Path .\C_evidence\c_reproduction_log.txt
   python --version
   python .\C_evidence\07_g4_stats_audit.py
   Write-Host "PythonExitCode: $LASTEXITCODE"
   Stop-Transcript
   ```

   **Do not publish** an unredacted transcript that contains personal Windows usernames, absolute paths, or other local information.

This README describes the **original C-side analysis layout**. It does not imply that the above commands have been independently rerun from a fresh clone; anyone making that claim must record their own successful execution.

Maintain the scripts in this repository's code/audits directory. For an acceptance rerun, project_source is a read-only source/record snapshot and C_evidence contains generated review outputs; neither is an additional editable project framework. A's acceptance work files are isolated under runs/_cache/pr4_review/, with plotting dependencies outside the frozen environment. The fixed experimental configuration remains configs/study.json.

## Results snapshot and interpretation

| Training condition | Mean top-1 accuracy | Same-seed gain vs real-only |
| --- | ---: | ---: |
| Real Only | 71.44% | — |
| Random 50 | 75.08% | +3.640 pp |
| CLIP 50 | 75.28% | +3.840 pp |
| Random 150 | 75.06% | +3.617 pp |
| CLIP 150 | 75.26% | +3.820 pp |

The separate, **single-run** CLIP zero-shot reference achieved 88.21%; it is a different model/inference approach and is not part of the 15-classifier paired-seed summary. CLIP selection exceeded Random by about 0.20 percentage points at either retained count; retaining 150 rather than 50 per class added essentially no mean benefit in this study. The three-seed SDs are **not confidence intervals** and these small contrasts should not be presented as statistically established improvements.

Recorded G3 production stages plus the G4 evaluation sum to approximately **48.14 minutes**. This is a sum of selected stage invocation times, **not total project wall-clock or development time**. Candidate generation includes recorded failures/retries, and its shared pool must not be charged repeatedly to each classifier.

## Independent check scope and pending external evidence

- **C checks passed locally:** G3/G4 archive checksum checks; saved CLIP selection consistency; G4 stored-prediction counts, hashes, IDs, class balance and top-1 recomputation; three-seed summaries and paired contrasts; per-class statistics; cost-record reconciliation; frozen-source/metadata and fairness checks. The earlier G3 selection and toy-statistics tests were submitted separately in PR #2.
- **Not reproduced by C:** original generation, CLIP inference, full training, independent GPU inference, or source-data relabeling. The training-fairness script checks saved records and source code, not execution-time numerical equivalence.
- **Related B finding:** the reported CPU-versus-sealed-GPU ResNet feature discrepancy exceeded the predeclared strict absolute threshold; see [`../../../report/B_v04_review.md`](../../../report/B_v04_review.md) for details. Do not silently count this as a successful strict cross-platform feature reproduction.
- **Human evidence:** the planned blinded two-annotator label audit and non-author usability trial should only be reported as completed if the relevant people actually perform them and provide records.

Please keep the original model/configuration seal and G4 release artifacts unchanged. C's outputs are downstream audit/analysis files, not a replacement official evaluation.
