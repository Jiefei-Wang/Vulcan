python -m scripts.ML.train_posaware ^
      --epochs 15 ^
      --batch-size 32 ^
      --range-min 5 ^
      --range-max 40 ^
      --margin 0.1 ^
      --sampling-strategy top ^
      --mine-batch-size 128 ^
      --max-training-samples 20000