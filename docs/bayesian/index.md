# Dual hNPE--hNDE

The dual model freezes five learned artifacts:

| Artifact | Role | Normalized object |
|---|---|---|
| $q_\phi(\theta\mid x)$ | parameter proposal | conditional posterior-flow proposal |
| $\widehat r_{\rm P}(\theta;x)$ | posterior correction | $q_\phi\widehat r_{\rm P}$, after proposal/design reweighting |
| $q_\eta(x\mid\theta)$ | observation proposal | conditional flow in observation space |
| $\widehat r_{\rm C}(x;\theta)$ | likelihood residual | corrects $q_\eta$ toward simulator data |
| $\widehat Z_{\rm C}(\theta)$ | conditional partition function | $q_\eta\widehat r_{\rm C}/\widehat Z_{\rm C}$ |

The posterior-side route is efficient and amortized. The likelihood-side route
is generative, supports alternative priors, and enables evidence, predictive,
and selection calculations. Their normalized weights must agree in the
population limit.

For a defensive parameter proposal

$$
g_\epsilon(\theta\mid x_o)
=(1-\epsilon)q_\phi(\theta\mid x_o)+\epsilon\rho(\theta),
$$

the posterior residual classifier must be trained against that same proposal
definition. Changing the inference proposal without changing the correction
changes the target.

The native trainer, portable artifact loader, and all inference calculations
are exported from `hnsbi.bayes`. Importing that namespace remains NumPy-only;
Torch, ONNX, and ONNX Runtime are resolved only when a corresponding training,
export, or runtime operation is requested.
