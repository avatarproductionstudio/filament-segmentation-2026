# Solar Filament Segmentation Challenge 2026 — Final Submission

**Team:** APS Team (Kaggle account `apsteam`)
**Selected final submission:** Kaggle ref **58252739**, public Panoptic Quality **0.38**
**Competition:** [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026) (IEEE BigData 2026 / AURA-NSO)

## Contents
- `pipeline.ipynb` — the entire prediction pipeline in one notebook: loads the
  weights from this repo, runs inference over the 180 test images, writes
  `submission.csv`, and validates the output (fail-closed).
- `requirements.txt` — packages and versions used by the pipeline.
- `weights/yolo26x-seg-solar-filament-2048-1109.pt` — the model checkpoint
  (stored via Git LFS), loaded by the notebook; no re-training is needed.

## Method
Single-stage instance segmentation at native resolution. A YOLO26x-seg model
is run once over each 2048×2048 GONG H-alpha test image
(`imgsz=2048, conf=0.25, iou=0.0, max_det=100`). Predicted masks are made
mutually disjoint by score-ordered overlap removal (the evaluator rejects
instance sets with multi-covered pixels), masks under 5 px are dropped, and
each instance is encoded as a compressed COCO RLE row
(`filament_id,segmentation_rle`). The selected run produced 1,435 rows
covering 179/180 test images.

The shipped configuration is the measured optimum of the sweeps we ran.
Detector→refiner cascades, test-time augmentation, output merging, threshold
calibration and fine-tuning variants were all built and evaluated on a
held-out local split and rejected for failing to beat this configuration;
the accompanying technical report gives the full measured record, including
the negative results.

## Model provenance
The checkpoint is the public model
[`hdjojo/yolo26x-seg-solar-filament`](https://www.kaggle.com/models/hdjojo/yolo26x-seg-solar-filament)
(PyTorch, default v1) by **HDJoJo**, trained on the 707 competition training
images only. It is redistributed here because the competition's final
submission rules require the weights to be stored in the repository and
loaded by the notebook. Our contribution is the measured configuration,
evaluation protocol, and selection process described in the report.

## Reproduce
```bash
pip install -r requirements.txt
# place the 180 competition test JPEGs in a folder, then:
FILAMENT_TEST_DIR=/path/to/test_images jupyter notebook pipeline.ipynb
```
On Kaggle, the competition data mount is picked up automatically. Output is
`submission.csv` plus a `run_report.json` validator summary.
