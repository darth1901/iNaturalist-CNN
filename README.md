# iNaturalist CNN

CS6910 Assignment 2. Image classification on the iNaturalist-12K dataset
(10 classes, ~10K training images), in two parts:

- **Part A:** a configurable CNN trained from scratch, tuned with a Weights &
  Biases hyperparameter sweep.
- **Part B:** fine-tuning an ImageNet-pretrained ResNet50 on the same data.

## Results

| | Part A (from scratch) | Part B (fine-tuned ResNet50) |
|---|---|---|
| Test accuracy | **0.39** | **0.845** |
| Initialisation | random | ImageNet-pretrained |
| Best config | 32 base filters, `double` filter org, 5x5 kernels, GELU, 128 dense neurons, dropout 0.3, augmentation on, Adam, lr 3e-4 | full fine-tuning (no frozen layers), AdamW, lr 2.3e-5, batch size 64, augmentation on |

The gap between the two is a representation-quality gap rather than a tuning
gap. A network trained from scratch on ~10K images cannot learn features that
separate 10 broad, high-variance biological classes, and its accuracy saturates
near 40% regardless of tuning. A pretrained backbone arrives with features
already learned from 1.2M ImageNet images.

## Repository structure

```
.
├── partA/
│   └── conv_nn.ipynb        # from-scratch CNN: model, sweep, test eval, interpretability
├── partB/
│   └── finetune.ipynb       # pretrained ResNet50: freezing strategies, fine-tuning, test eval
└── README.md
```

## Running Part A

`partA/conv_nn.ipynb` contains, in order:

1. **Data pipeline.** Download, cleanup, and a one-time pre-resize of all images
   to 224x224 on disk. This last step matters: resizing inside the transform
   pipeline re-decodes the original full-size JPEGs on every epoch of every run,
   which starves the GPU on a 2-core Colab box. Pre-resizing once removes that
   bottleneck.
2. **`ConfigurableCNN`.** Five conv blocks (`conv -> batchnorm (optional) ->
   activation -> maxpool`) followed by a dense layer and a 10-way output. Number
   of filters, filter organisation (`same` / `double` / `halve`), kernel size,
   activation, dense width, batchnorm, and dropout are all configurable. The
   flatten dimension is inferred with a dummy forward pass, so the model adapts
   to any input resolution.
3. **Sweep.** Random search with Hyperband early termination (`min_iter=3`),
   maximising validation accuracy. Learning rate is sampled log-uniformly, since
   meaningful steps in lr are multiplicative.
4. **Test evaluation** of the best config, plus a 10x3 prediction grid.
5. **Interpretability.** First-layer filter visualisation and guided
   backpropagation on the CONV5 channels.

To launch the sweep:

```python
sweep_id = wandb.sweep(sweep_config, project="inaturalist-cnn")
wandb.agent(sweep_id, sweep_run, count=30)
```

## Running Part B

`partB/finetune.ipynb` reuses the same data pipeline and training scheme. The
differences are confined to three places:

- **`build_model`** loads `ResNet50_Weights.IMAGENET1K_V2` and replaces the
  1000-class head with a fresh 10-class `Linear` layer. Input images are resized
  to 224x224 and normalised with ImageNet statistics to match what the backbone
  expects.
- **Freezing strategies.** Three are implemented and swept:
  `feature_extract` (freeze everything except the new head),
  `freeze_up_to` (freeze the early layers, fine-tune the later ones plus the
  head), and `full_finetune` (nothing frozen). The optimiser is given only
  parameters with `requires_grad=True`.
- **Learning rate range** is an order of magnitude lower than Part A
  (1e-5 to 1e-3). Fine-tuning adapts weights that are already good, so a high
  learning rate would destroy the pretrained features rather than refine them.

The best result came from full fine-tuning at a very small learning rate
(2.3e-5), which is the principled way to run it: unfreeze everything, but nudge
it gently.

## WandB Report
- [Part A](https://wandb.ai/parthd1901-bits-pilani/inaturalist-cnn)
- [Part B](https://wandb.ai/parthd1901-bits-pilani/inaturalist-finetuned)
