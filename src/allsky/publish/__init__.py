"""The documents the public sky page reads about the all-sky model.

One command, :mod:`allsky.cli.publish`, writes them all; the builders live here,
one per document, torch-free where the document needs no forward pass:

- :mod:`allsky.publish.encoding` — the shared byte contract (schema ids, the
  publish stamp, the compact strict-JSON writer);
- :mod:`allsky.publish.frame` — ``frame.json`` and its three images, the one
  builder that runs forward passes (the occlusion sweep and the
  counterfactuals on the latest scored frame);
- :mod:`allsky.publish.timeline` — ``timeline.json``, the watch's block
  predictions of the last days against the clear-sky reference;
- :mod:`allsky.publish.model_card` — ``model.json``, the served checkpoints
  against the controls and the baselines on the held-out test days;
- :mod:`allsky.publish.dataset` — what the builders read from the prepared
  dataset (the training split's highest solar elevation).

The contract is documented in ``docs/allsky-site.md``.
"""
