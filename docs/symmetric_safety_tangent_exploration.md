# Symmetric Safety-Tangent Exploration

This update replaces isotropic Gaussian action perturbations with geometry-guided directional noise:

\[ \epsilon_G(s) \sim \mathcal{N}(0, \Sigma_G(s)) \]

\[ \Sigma_G(s)=\sigma_g(s)^2 e_g e_g^T + \sigma_t(s)^2 e_t e_t^T + \sigma_n(s)^2 e_n e_n^T \]

- `e_g`: goal direction.
- `e_n`: obstacle outward normal direction.
- `e_t`: obstacle tangent direction.
- `sigma_t` increases near obstacle; `sigma_n` is reduced near obstacle.
- both positive and negative tangent candidates are explicitly sampled.

Design intent:
- before filter activation, directional noise provides proactive safe exploration;
- near obstacle, tangent exploration rises continuously (not as a hard switch);
- when filter activates, directional perturbations help pull raw policy toward low projection-cost manifolds;
- symmetric tangent candidates mitigate single-mode (upper-only or lower-only) collapse;
- uniform mixed softmax weights reduce early one-hot collapse from critic bias.
