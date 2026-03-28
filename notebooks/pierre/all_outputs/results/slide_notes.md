# MTL vs Baseline — Key results

- Dataset: Forest vs Residential (images 64x64)
- Training on Apple M4 (MPS); 10 epochs per model

## Metrics
{
  "baseline": {
    "acc": 0.7333333333333333,
    "cm": [
      [
        29,
        31
      ],
      [
        1,
        59
      ]
    ]
  },
  "mtl": {
    "acc": 0.5833333333333334,
    "cm": [
      [
        11,
        49
      ],
      [
        1,
        59
      ]
    ]
  },
  "softshare": {
    "acc": 0.5083333333333333,
    "cm": [
      [
        1,
        59
      ],
      [
        0,
        60
      ]
    ]
  }
}

## Takeaways
- Baseline CNN: high val accuracy on validation set, OOD acc ~0.73, but struggles on Forest subset (confusion with Residential).
- MTL model: similar validation acc but lower OOD overall (0.58) — seems to underperform on Forest more than baseline.
- SoftShare: also shows high val acc but very low Forest OOD accuracy (near 0), indicating overfitting to Residential subdomains or dataset shift.
- Dense/Medium OOD subgroups achieve high accuracy across models — Forest appears to be the problematic domain.

## Speaker notes
- Explain the difference between validation accuracy (in-domain) and OOD accuracy (external set).
- Point to confusion matrices: Forest (true class 0) is often predicted as Residential (1) — indicate domain gap and possible label/visual ambiguity.
- Hypotheses: class imbalance, covariate shift, or background features dominating; recommend visual inspection and balanced sampling.
- Next steps: collect more Forest examples, try domain adaptation / augmentation focused on Forest, or fine-tune with OOD samples.
