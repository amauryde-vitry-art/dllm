```latex
\subsection{Baseline Features}
\label{subsec:baseline}

We define a set of scalar features extracted from the diffusion trajectory of each generated sample. Let $H^{(i)}_{t,d}$ denote the entropy of the model's predictive distribution at diffusion step $t \in \{1, \dots, T\}$ and token position $d \in \{1, \dots, D\}$ for sample $i$, and let $\ell^{(i)}_{t,d} = \log p_\theta(x_d \mid x_t)$ denote the log-probability assigned to the final token at that position. We further define the binary mask $m^{(i)}_{t,d} \in \{0,1\}$ indicating whether token $d$ is still masked at step $t$.

\subsubsection{Mean Features}

\paragraph{Mean Entropy.}
The global mean entropy aggregates uncertainty across all steps and positions:
\begin{equation}
    \overline{H}^{(i)} = \frac{1}{T \cdot D} \sum_{t=1}^{T} \sum_{d=1}^{D} H^{(i)}_{t,d}
\end{equation}
A high value indicates sustained uncertainty throughout the generation process, which may signal hallucinated content.

\paragraph{Mean Masked Entropy.}
To focus on positions where the model has not yet committed to a token, we restrict the average to masked positions:
\begin{equation}
    \overline{H}_{\mathrm{masked}}^{(i)} = \frac{\sum_{t,d} m^{(i)}_{t,d} \cdot H^{(i)}_{t,d}}{\sum_{t,d} m^{(i)}_{t,d}}
\end{equation}
This avoids diluting the signal with already-revealed tokens whose entropy is trivially low.

\paragraph{Mean Entropy at Unmasking.}
For each step $t$, we identify the set of tokens that are unmasked at exactly step $t$ (i.e.\ $m^{(i)}_{t-1,d}=1$ and $m^{(i)}_{t,d}=0$). The entropy at the moment of unmasking reflects the model's confidence when making its decision:
\begin{equation}
    \overline{H}_{\mathrm{unmask}}^{(i)} = \frac{1}{D} \sum_{t=1}^{T} \sum_{d : \Delta m_{t,d}=1} H^{(i)}_{t,d}
\end{equation}
where $\Delta m_{t,d} = m^{(i)}_{t-1,d} - m^{(i)}_{t,d}$.  A high value here indicates that the model is uncertain even at the step where it commits to a token.

\paragraph{Mean Log-Probability.}
\begin{equation}
    \overline{\ell}^{(i)} = \frac{1}{T \cdot D} \sum_{t=1}^{T} \sum_{d=1}^{D} \ell^{(i)}_{t,d}
\end{equation}
This is a straightforward confidence measure: lower (more negative) values reflect less confident predictions on average.

\subsubsection{Variance Features}

\paragraph{Variance of Entropy Across Steps (per token).}
For each token position $d$, we compute the temporal variance of entropy and then average over positions:
\begin{equation}
    \mathrm{VarEntropy}^{(i)} = \frac{1}{D} \sum_{d=1}^{D} \mathrm{Var}_t\!\left( H^{(i)}_{t,d} \right)
\end{equation}
A high temporal variance indicates that the model's uncertainty for a given position fluctuates strongly across diffusion steps, suggesting instability in the generation process.

\paragraph{Variance of Entropy Across Tokens (per step).}
Conversely, for each step we compute the spatial variance across token positions:
\begin{equation}
    \mathrm{VarEntropy}_{\mathrm{tokens}}^{(i)} = \frac{1}{T} \sum_{t=1}^{T} \mathrm{Var}_d\!\left( H^{(i)}_{t,d} \right)
\end{equation}
A high spatial variance means that at a given step, the model is very uncertain about some tokens and confident about others, reflecting heterogeneous difficulty across the sequence.

\paragraph{Masked Variance Across Steps (per token).}
To avoid contamination from steps where a token is already revealed, we restrict the temporal variance to masked steps only:
\begin{equation}
    \mathrm{VarEntropy}_{\mathrm{masked}}^{(i)} = \frac{1}{|\mathcal{D}|}\sum_{d \in \mathcal{D}} \mathrm{Var}\!\left( \left\{ H^{(i)}_{t,d} : m^{(i)}_{t,d} = 1 \right\} \right)
\end{equation}
where $\mathcal{D}$ is the set of positions with at least two masked steps.

\paragraph{Masked Variance Across Tokens (per step).}
At each step, we compute the variance of entropy among the tokens that are still masked, then average over steps:
\begin{equation}
    \mathrm{VarEntropy}_{\mathrm{masked,tokens}}^{(i)} = \frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}} \mathrm{Var}\!\left( \left\{ H^{(i)}_{t,d} : m^{(i)}_{t,d} = 1 \right\} \right)
\end{equation}
where $\mathcal{T}$ is the set of steps with at least two masked tokens. This captures how heterogeneous the difficulty is among remaining positions at each step. An increase over time may indicate growing disagreement in the model.

\paragraph{Variance of Entropy at Unmasking.}
\begin{equation}
    \mathrm{Var}(H_{\mathrm{unmask}})^{(i)} = \mathrm{Var}_t\!\left( \bar{H}^{(i)}_{\mathrm{unmask},t} \right)
\end{equation}
where $\bar{H}^{(i)}_{\mathrm{unmask},t}$ is the mean entropy of tokens unmasked at step $t$. High variance suggests the model alternates between confident and uncertain unmasking decisions.

\paragraph{Variance of Log-Probabilities Across Steps (per token).}
\begin{equation}
    \mathrm{VarLogProb}^{(i)} = \frac{1}{D} \sum_{d=1}^{D} \mathrm{Var}_t\!\left( \ell^{(i)}_{t,d} \right)
\end{equation}
A high value indicates that the model's confidence for individual token positions oscillates across diffusion steps.

\subsubsection{Dynamic Features}

These features characterise the \emph{temporal evolution} of entropy and log-probabilities along the diffusion trajectory.

\paragraph{Discrete Rate of Change (Tau).}
We estimate the local rate of change of the step-averaged entropy using a finite-difference scheme with window $w$:
\begin{equation}
    \tau^{(i)}_k = \frac{\bar{H}^{(i)}_{(k+1)w} - \bar{H}^{(i)}_{kw}}{w}, \qquad \bar{H}^{(i)}_t = \frac{1}{D}\sum_d H^{(i)}_{t,d}
\end{equation}
We then extract $\mathrm{MeanTau}^{(i)} = \mathbb{E}_k[\tau^{(i)}_k]$ and $\mathrm{VarTau}^{(i)} = \mathrm{Var}_k(\tau^{(i)}_k)$. The mean tau captures the overall trend (decreasing entropy = convergence), while the variance captures irregularity of this convergence. The same is computed symmetrically for log-probabilities.

\paragraph{Exponential Decay Fit ($\alpha$, $\beta$).}
We model the per-token log-entropy trajectory as an exponential decay:
\begin{equation}
    \ln H^{(i)}_{t,d} \approx \alpha_d \cdot t + \beta_d
\end{equation}
The slope $\alpha_d < 0$ characterises the decay rate: a slow decay (small $|\alpha|$) may indicate difficulty in resolving a token. We report:
\begin{equation}
    \alpha^{(i)}_H = \frac{1}{D}\sum_d \alpha_d, \qquad \beta^{(i)}_H = \frac{1}{D}\sum_d \beta_d
\end{equation}
The same procedure is applied to log-probabilities to obtain $\alpha^{(i)}_\ell$ and $\beta^{(i)}_\ell$.

\paragraph{Top-$k$ Averaged Exponential Decay.}
To focus on the most uncertain tokens, we select the $k$ positions with highest mean entropy, average their entropy trajectories, and fit:
\begin{equation}
    \ln\!\left(\frac{1}{k}\sum_{d \in \mathrm{top}\text{-}k} H^{(i)}_{t,d}\right) \approx \alpha^{(i)}_{H,k} \cdot t + \beta^{(i)}_{H,k}
\end{equation}
This provides a more robust estimate of the decay rate on the ``hardest'' tokens.

\paragraph{Parametric Trajectory Fit ($C$, $\tau$, $m$).}
We fit the averaged entropy trajectory to a parametric model capturing a rise-then-decay shape:
\begin{equation}
    \bar{H}^{(i)}_{\mathrm{top\text{-}k}}(t) \approx C \cdot t \cdot \exp\!\left(-\frac{t - m}{\tau}\right)
\end{equation}
where $C$ controls the amplitude, $m$ the peak location, and $\tau$ the decay timescale. A large $\tau$ indicates slow convergence, while $m$ localises where the model is most uncertain during the process.

\subsubsection{Shape Features on the Masked Variance Curve}

Let $V^{(i)}(t) = \mathrm{Var}\!\left(\{H^{(i)}_{t,d} : m^{(i)}_{t,d}=1\}\right)$ be the inter-token variance of entropy among masked positions at step $t$. This curve characterises the evolving heterogeneity of the model's uncertainty. We extract:

\begin{itemize}
    \item \textbf{AUC}: $\int_0^T V^{(i)}(t)\,dt$ (total cumulated dispersion);
    \item \textbf{Max}: $\max_t V^{(i)}(t)$ (peak heterogeneity);
    \item \textbf{Argmax}: $\arg\max_t V^{(i)}(t)$ (step of peak heterogeneity);
    \item \textbf{Skewness}: asymmetry of the $V^{(i)}(t)$ distribution;
    \item \textbf{Kurtosis}: peakedness of the $V^{(i)}(t)$ distribution;
    \item \textbf{Mean Curvature}: $\frac{1}{T}\sum_t \frac{|V''(t)|}{(1 + V'(t)^2)^{3/2}}$, measuring how sharply the curve bends on average.
\end{itemize}

These shape descriptors capture qualitative differences in how the model resolves uncertainty: a sharp narrow peak suggests abrupt decision-making, while a flat broad curve suggests gradual, distributed resolution.
```
