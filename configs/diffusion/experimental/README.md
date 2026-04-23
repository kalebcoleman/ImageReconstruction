# Experimental Diffusion Recipes

These archived recipes preserve the older strict `64x64` RGB ADM parity
protocol that previously forced every dataset through the same geometry and
backbone.

They remain available for comparison or historical reproduction, but they are
not the default final-study configs anymore.

The default [`run_parity_suite.py`](/Users/itzjuztmya/Kaleb/ImageReconstruction/run_parity_suite.py)
workflow never selects these archived recipes unless you intentionally override
`--config-dir`.

Archived full-study recipes:

- `mnist_64.yaml`
- `fashion_64.yaml`
- `cifar10_64.yaml`
- `imagenet_64.yaml`

Archived smoke recipes live under `../smoke/experimental/`.
