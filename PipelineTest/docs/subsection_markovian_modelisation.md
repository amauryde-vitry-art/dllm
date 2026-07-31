```latex
\subsection{Markovian Modelisation}
\label{subsec:markovian}

We develop a simplified stochastic model that motivates the functional forms used to extract dynamic features from the diffusion trajectory. The key insight is that each token position, viewed marginally, behaves approximately as an absorbing Markov process converging to a single state — and the entropy of this process decays in a characteristic way that we can exploit for hallucination detection.

\subsubsection{Single-Token Model}

\paragraph{Setup.}
Consider a discrete stochastic process on $N$ states $\{1, 2, \dots, N\}$ (representing the vocabulary) associated to a single token position $d$. We model the predictive distribution $p_\theta(\cdot \mid x_t)$ at position $d$ as converging toward a single dominant state $i^* \in \{1,\dots,N\}$ as the diffusion progresses. Specifically, for $t \gg \tau$:
\begin{align}
    P(i^*, t) &= 1 - \varepsilon(t), \\
    P(j, t) &= \frac{\varepsilon(t)}{N-1}, \qquad \forall\, j \neq i^*,
\end{align}
where the residual probability mass satisfies:
\begin{equation}
    \varepsilon(t) \approx C \, e^{-t/\tau},
\end{equation}
with $C > 0$ an amplitude and $\tau > 0$ a characteristic timescale controlling how fast the model converges to its final prediction. The normalisation $\sum_j P(j,t) = 1$ is satisfied by construction.

\paragraph{Interpretation.}
This model captures the intuition that during the reverse diffusion process, the model progressively commits to a token at each position. The timescale $\tau$ reflects the ``difficulty'' of the position: easy tokens (common words, deterministic continuations) have small $\tau$, while hard tokens (ambiguous, hallucinated) have large $\tau$.

\subsubsection{Entropy of a Single Process}

\begin{proposition}
For $t \gg \tau$ (i.e.\ $\varepsilon(t) \ll 1$), the Shannon entropy of the single-token process satisfies:
\begin{equation}
    \boxed{S(t) \;\approx\; C\,e^{-t/\tau}\left(\frac{t}{\tau} + \ln\frac{N-1}{C} + 1 \right)}
\end{equation}
In particular, for $t/\tau \gg \ln((N-1)/C)$:
\begin{equation}
    S(t) \;\approx\; \frac{C}{\tau}\, t\, e^{-t/\tau}.
\end{equation}
\end{proposition}

\begin{proof}
The Shannon entropy is:
\begin{align}
    S(t) &= -P(i^*,t)\ln P(i^*,t) - \sum_{j \neq i^*} P(j,t)\ln P(j,t) \\
    &= -(1-\varepsilon)\ln(1-\varepsilon) - (N-1)\cdot\frac{\varepsilon}{N-1}\cdot\ln\frac{\varepsilon}{N-1} \\
    &= -(1-\varepsilon)\ln(1-\varepsilon) - \varepsilon\ln\varepsilon + \varepsilon\ln(N-1).
\end{align}
For $\varepsilon \ll 1$, using $\ln(1-\varepsilon) \approx -\varepsilon - \varepsilon^2/2 \approx -\varepsilon$:
\begin{align}
    -(1-\varepsilon)\ln(1-\varepsilon) &\approx (1-\varepsilon)\cdot\varepsilon \approx \varepsilon.
\end{align}
Substituting $\varepsilon = C\,e^{-t/\tau}$ so that $\ln\varepsilon = \ln C - t/\tau$:
\begin{align}
    S(t) &\approx \varepsilon - \varepsilon\ln\varepsilon + \varepsilon\ln(N-1) \\
    &= \varepsilon\bigl(1 - \ln C + t/\tau + \ln(N-1)\bigr) \\
    &= C\,e^{-t/\tau}\left(\frac{t}{\tau} + \ln\frac{N-1}{C} + 1\right).
\end{align}
\end{proof}

\paragraph{Key properties of $S(t)$.}
\begin{itemize}
    \item $S(t)$ has a \emph{rise-then-decay} shape: it increases (due to the linear factor $t/\tau$) before being dominated by the exponential decay $e^{-t/\tau}$.
    \item The peak occurs near $t^* \approx \tau$ (precisely when $dS/dt = 0$).
    \item The \emph{logarithm} of the entropy for large $t$ is approximately linear:
    \begin{equation}
        \ln S(t) \approx -\frac{t}{\tau} + \ln t + \mathrm{const},
    \end{equation}
    which is dominated by the linear term $-t/\tau$ for $t \gg \tau$.
\end{itemize}

\subsubsection{Average Entropy of $K$ Identical Processes}

\begin{proposition}
Consider $K$ independent token positions, each following the same model with parameters $(C_k, \tau, N)$ sharing the same timescale $\tau$. The average entropy:
\begin{equation}
    \langle S \rangle(t) = \frac{1}{K}\sum_{k=1}^K S_k(t)
\end{equation}
satisfies:
\begin{equation}
    \boxed{\langle S \rangle(t) \;\approx\; \bar{C}\,e^{-t/\tau}\left(\frac{t}{\tau} + \ln\frac{N-1}{\bar{C}} + 1\right)}
\end{equation}
where $\bar{C} = \frac{1}{K}\sum_k C_k$.
\end{proposition}

\begin{proof}
Since all processes share the same $\tau$:
\begin{align}
    \langle S \rangle(t) &= \frac{1}{K}\sum_{k=1}^K C_k\,e^{-t/\tau}\left(\frac{t}{\tau} + \ln\frac{N-1}{C_k} + 1\right) \\
    &= e^{-t/\tau}\left[\bar{C}\left(\frac{t}{\tau}+1\right) + \frac{1}{K}\sum_k C_k\ln\frac{N-1}{C_k}\right].
\end{align}
For $t/\tau \gg 1$, the term $\bar{C}\cdot t/\tau$ dominates, giving:
\begin{equation}
    \langle S \rangle(t) \approx \frac{\bar{C}}{\tau}\,t\,e^{-t/\tau},
\end{equation}
which retains the same functional form as the single-process entropy.
\end{proof}

\subsubsection{Average Entropy with Heterogeneous Timescales}

In practice, different token positions have different convergence rates. We now consider $K$ processes with distinct timescales $\tau_k$.

\begin{proposition}
The average entropy of $K$ processes with heterogeneous timescales $\{\tau_k\}_{k=1}^K$ is:
\begin{equation}
    \langle S \rangle(t) = \frac{1}{K}\sum_{k=1}^K C_k\,e^{-t/\tau_k}\left(\frac{t}{\tau_k} + \ln\frac{N-1}{C_k} + 1\right).
\end{equation}
This is a \emph{mixture of asymmetric decays} with no closed-form simplification in general.
\end{proposition}

\paragraph{Approximations.}

\begin{enumerate}
    \item \textbf{Late-time dominance.} For $t$ large, the sum is dominated by the slowest-decaying component (largest $\tau_k$). Let $\tau_{\max} = \max_k \tau_k$ and let $\mathcal{K}_{\max}$ be the set of indices achieving or approaching this maximum. Then:
    \begin{equation}
        \langle S \rangle(t) \;\underset{t \to \infty}{\sim}\; \frac{1}{K}\sum_{k \in \mathcal{K}_{\max}} C_k\,\frac{t}{\tau_{\max}}\,e^{-t/\tau_{\max}}.
    \end{equation}
    The effective decay rate of the \emph{average} entropy reflects the hardest tokens, not the typical ones.

    \item \textbf{Log-linear approximation.} If the $\tau_k$ are narrowly distributed around a mean $\bar{\tau}$ with small variance $\sigma_\tau^2$, a second-order expansion gives:
    \begin{equation}
        \ln\langle S \rangle(t) \approx -\frac{t}{\bar{\tau}} + \frac{\sigma_\tau^2\,t^2}{2\bar{\tau}^4} + \ln t + \mathrm{const}.
    \end{equation}
    The quadratic correction means that a simple linear fit $\ln S \approx \alpha t + \beta$ will yield:
    \begin{equation}
        \alpha \approx -\frac{1}{\bar{\tau}} + \frac{\sigma_\tau^2\,t}{2\bar{\tau}^4}\bigg|_{\text{effective}},
    \end{equation}
    i.e.\ the fitted slope $\alpha$ reflects a combination of the mean timescale and its dispersion.

    \item \textbf{Parametric fit.} The functional form:
    \begin{equation}
        \langle S \rangle(t) \approx C_{\mathrm{eff}} \cdot (t - m) \cdot \exp\!\left(-\frac{t-m}{\tau_{\mathrm{eff}}}\right)
    \end{equation}
    provides a good empirical approximation, where $\tau_{\mathrm{eff}}$ is an effective timescale influenced by the slowest processes, and $m$ captures the delay before the entropy peaks.
\end{enumerate}

\subsubsection{Connection to Dynamic Features}

The Markovian model directly motivates the dynamic features extracted from the diffusion trajectories (cf.\ Section~\ref{subsec:baseline}).

\paragraph{Exponential decay parameters ($\alpha$, $\beta$).}
From the model, $\ln S(t) \approx -t/\tau + \ln t + \mathrm{const}$, which for large $t$ is approximately linear. The features:
\begin{equation}
    \ln H^{(i)}_{t,d} \approx \alpha_d \cdot t + \beta_d
\end{equation}
directly estimate:
\begin{equation}
    \alpha_d \approx -\frac{1}{\tau_d}, \qquad \beta_d \approx \ln C_d + \ln(N-1).
\end{equation}
A hallucinated sample is expected to have \textbf{smaller} $|\alpha|$ (i.e.\ $\alpha$ closer to zero), reflecting a \textbf{larger} effective $\tau$ — slower convergence, persistent uncertainty.

\paragraph{Top-$k$ averaged decay ($\alpha_{H,k}$, $\beta_{H,k}$).}
By selecting the $k$ tokens with highest mean entropy, we effectively select those with the largest $\tau_k$. The fitted slope:
\begin{equation}
    \alpha_{H,k} \approx -\frac{1}{\tau_{\mathrm{eff,top\text{-}k}}}
\end{equation}
is more sensitive to hard tokens and thus more discriminative for hallucination detection.

\paragraph{Parametric trajectory fit ($C$, $\tau$, $m$).}
The fitted model:
\begin{equation}
    \bar{H}_{\mathrm{top\text{-}k}}(t) \approx C \cdot t \cdot \exp\!\left(-\frac{t-m}{\tau}\right)
\end{equation}
directly implements the functional form derived in the single-process case. The parameters have clear interpretations:
\begin{itemize}
    \item $C$: overall amplitude of uncertainty — higher $C$ means the model was initially more confused;
    \item $\tau$: effective convergence timescale — \textbf{larger $\tau$ signals hallucination};
    \item $m$: peak location — a later peak ($m$ large) indicates the model took longer to reach maximum entropy before beginning to converge.
\end{itemize}

\paragraph{Mean Tau (finite differences).}
The discrete rate of change:
\begin{equation}
    \tau^{(i)}_k = \frac{\bar{H}_{(k+1)w} - \bar{H}_{kw}}{w}
\end{equation}
approximates $d\bar{H}/dt$ at discrete intervals. From the model:
\begin{equation}
    \frac{dS}{dt} \approx \frac{C}{\tau}\,e^{-t/\tau}\left(1 - \frac{t}{\tau}\right),
\end{equation}
so $\mathrm{MeanTau}$ captures the average derivative, which should be more negative (faster decay) for non-hallucinated samples and closer to zero for hallucinated ones.

\paragraph{Variance of masked entropy across tokens.}
In the heterogeneous-$\tau$ regime, at each step $t$ the inter-token variance of entropy reflects the \emph{dispersion} of timescales:
\begin{equation}
    \mathrm{Var}_d(S_d(t)) \approx \mathrm{Var}_d\!\left(C_d\,e^{-t/\tau_d}\cdot\frac{t}{\tau_d}\right).
\end{equation}
This variance has a characteristic rise-then-decay shape (peaking when the spread in convergence states is maximal), which motivates the shape features (AUC, max, argmax, skewness, kurtosis, curvature) extracted from the $V^{(i)}(t)$ curve. For hallucinated samples, the heterogeneity in $\tau_k$ values leads to a broader and more skewed variance curve.

\subsubsection{Summary}

The Markovian model provides a principled justification for the dynamic features:

\begin{center}
\begin{tabular}{lll}
\toprule
\textbf{Feature} & \textbf{Model parameter} & \textbf{Hallucination signal} \\
\midrule
$\alpha_H$ (slope of $\ln H$) & $-1/\tau$ & $\alpha \to 0$ (slow decay) \\
$\beta_H$ (intercept of $\ln H$) & $\ln C + \ln(N-1)$ & large $\beta$ (high initial confusion) \\
$\tau_{\mathrm{eff}}$ (parametric fit) & effective timescale & large $\tau$ \\
$m$ (peak location) & entropy peak time & large $m$ (late peak) \\
$C$ (amplitude) & initial residual mass & large $C$ \\
MeanTau (finite diff.) & $\langle dS/dt \rangle$ & close to $0$ \\
Shape features on $V(t)$ & $\tau$-dispersion & broad, skewed curve \\
\bottomrule
\end{tabular}
\end{center}
```
