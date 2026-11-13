# Model

The ensemble classifier of the paper (Section 3) will be added here: a CNN–BiGRU network on the |E| envelope,
two LightGBM models on envelope descriptors and resonance parameters, and logistic stacking of their scores.
The candidate with the highest probability in each realization is selected.
