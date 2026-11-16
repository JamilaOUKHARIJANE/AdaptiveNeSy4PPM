Adaptive Weighting to Automatically Balance Neural and Symbolic Contributions in Neuro-Symbolic Predictive Process Monitoring
================

AdaptiveNeSy4PPM combines a neural next-activity predictor with process model (PM).  During suffix prediction, the system compute a new weight for every decoded step.  The
adaptive mode is designed to give more influence to the source that is more
reliable for the current prefix.

The implementation supports activity-only, beam search, and several PM
representations, including Declare and Petri-net/procedural models.

**Adaptive weighting** is controlled by boolean options passed to
``predict_evaluate``.  **The available signals are:**

``nn_entropy``
    Use ``1 - U_NN`` as a neural reliability factor, where ``U_NN`` is the
    normalized entropy of the neural probability distribution.  A confident
    neural distribution has a larger factor.

``pm_entropy``
    Use ``1 - U_PM`` as a PM reliability factor.

``prefix_support``
    Use the normalized frequency of the current prefix in the training log.
    An unseen prefix has support ``0``.

``prefix_sim``
    Use the best normalized Damerau-Levenshtein similarity between the current
    prefix and a prefix observed in the training data.

``pm_conformance``
    Use the current prefix's PM alignment/conformance score when the selected
    PM model provides one.  The exact score depends on the PM type: for
    example, it can be Declare compliance or Petri-net token-replay fitness.

An adaptive run that enables all available signal families can therefore be
configured with ``nn_entropy=True``, ``pm_entropy=True``,
``prefix_support=True``, ``prefix_sim=True``, and ``pm_conformance=True``.
The flags also make it possible to run signal ablations.
