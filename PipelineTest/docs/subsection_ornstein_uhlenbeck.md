```latex
\subsection{Ornstein--Uhlenbeck Modelling and AR(1) Discretisation}
\label{subsec:ou_ar1}

The Markovian model of Section~\ref{subsec:markovian} captures the \emph{deterministic} envelope of entropy decay, but the actual per-token entropy trajectories exhibit significant stochastic fluctuations around this envelope. To model both the drift toward convergence and the residual noise, we adopt a \emph{time-inhomogeneous Ornstein--Uhlenbeck} (OU) framework, which we then discretise exactly as an AR(1) process for practical estimation.

\subsubsection{Continuous-Time Model}

\paragraph{Setup.}
Let $X_{t,d}$ denote the entropy of the model's predictive distribution at diffusion step $t$ for token position $d$. We model $X_{t,d}$ as a time-inhomogeneous Ornstein--Uhlenbeck process:
\begin{equation}
    \boxed{dX_{t,d} = \mu_d(t)\bigl(\theta_d(t) - X_{t,d}\bigr)\,dt + \sigma_d(t)\,dW_{t,d}}
\end{equation}
where:
\begin{itemize}
    \item $\mu_d(t) > 0$ is the \textbf{mean-reversion rate}: how strongly $X_{t,d}$ is pulled toward equilibrium. In the context of the diffusion sampler, this quantifies the ``denoising strength'' at step $t$ for token $d$.
    \item $\theta_d(t)$ is the \textbf{long-term target}: the equilibrium entropy value (typically $\theta_d \to 0$ as the model converges to a deterministic prediction).
    \item $\sigma_d(t) > 0$ is the \textbf{volatility}: the stochastic noise injected at each step due to the coupling between token positions and the randomness of the reverse diffusion schedule.
    \item $W_{t,d}$ is a standard Wiener process (independent across tokens $d$).
\end{itemize}

\paragraph{Moment evolution.}
For the OU process, the conditional moments evolve as:
\begin{align}
    \frac{d}{dt}\mathbb{E}[X_{t,d}] &= \mu_d(t)\bigl(\theta_d(t) - \mathbb{E}[X_{t,d}]\bigr), \label{eq:ou_mean}\\
    \frac{d}{dt}\mathrm{Var}(X_{t,d}) &= -2\mu_d(t)\,\mathrm{Var}(X_{t,d}) + \sigma_d(t)^2. \label{eq:ou_var}
\end{align}
Equation~\eqref{eq:ou_mean} shows that the mean entropy decays exponentially toward $\theta_d$ with rate $\mu_d$ (consistent with the Markovian envelope $S(t) \sim e^{-t/\tau}$ from Section~\ref{subsec:markovian}, where $\mu_d \sim 1/\tau_d$). Equation~\eqref{eq:ou_var} shows that the variance reaches a \emph{steady state}:
\begin{equation}
    \mathrm{Var}_{\mathrm{ss}}(X_{t,d}) = \frac{\sigma_d(t)^2}{2\mu_d(t)},
\end{equation}
when the parameters are locally constant. This steady-state variance quantifies the residual ``jitter'' in the entropy trajectory.

\paragraph{Population-level statistics.}
Across $D$ token positions (with heterogeneous parameters), the inter-token variance of entropy:
\begin{equation}
    V(t) = \mathrm{Var}_d\bigl(X_{t,d}\bigr) = \frac{1}{D}\sum_{d=1}^D \bigl(X_{t,d} - \bar{X}_t\bigr)^2
\end{equation}
is \emph{not} simply the variance of a single OU process, but rather reflects the \emph{dispersion of the population} of OU processes with different parameters $(\mu_d, \theta_d, \sigma_d)$. Its evolution is governed by:
\begin{itemize}
    \item \textbf{Early phase} ($t$ small): most tokens are masked and have high entropy; the inter-token variance is relatively low (homogeneous uncertainty).
    \item \textbf{Middle phase}: tokens begin unmasking at different rates, creating a mixture of converged (low entropy) and unconverged (high entropy) positions $\Rightarrow$ the inter-token variance $V(t)$ \emph{increases}.
    \item \textbf{Late phase}: most tokens have converged, leaving only a few ``hard'' tokens with high entropy $\Rightarrow$ $V(t)$ \emph{decreases}.
\end{itemize}
This rise-then-decay of $V(t)$ is the same characteristic shape captured by the shape features in Section~\ref{subsec:baseline}.

\subsubsection{AR(1) Discretisation}

\paragraph{Exact discretisation.}
The exact solution of the OU SDE between discrete steps $t-1$ and $t$ (with $\Delta t = 1$) gives:
\begin{equation}
    X_{t,d} = e^{-\mu_d(t)} X_{t-1,d} + \theta_d(t)\bigl(1 - e^{-\mu_d(t)}\bigr) + \sigma_d(t)\sqrt{\frac{1 - e^{-2\mu_d(t)}}{2\mu_d(t)}}\;\varepsilon_{t,d}
\end{equation}
where $\varepsilon_{t,d} \sim \mathcal{N}(0,1)$ i.i.d. This is an \textbf{AR(1) process} with time- and position-dependent coefficients:
\begin{equation}
    \boxed{X_{t,d} = \varphi_{t,d}\, X_{t-1,d} + c_{t,d} + \sigma^\varepsilon_{t,d}\,\varepsilon_{t,d}}
\end{equation}
where:
\begin{align}
    \varphi_{t,d} &= e^{-\mu_d(t)} \in [0,1], \label{eq:phi_ou}\\
    c_{t,d} &= \theta_d(t)\bigl(1 - e^{-\mu_d(t)}\bigr), \label{eq:c_ou}\\
    \sigma^\varepsilon_{t,d} &= \sigma_d(t)\sqrt{\frac{1 - e^{-2\mu_d(t)}}{2\mu_d(t)}}. \label{eq:sigma_ou}
\end{align}

\paragraph{Interpretation of the AR(1) coefficients.}
\begin{itemize}
    \item $\varphi_{t,d}$ (\texttt{phi} in the code): measures the \textbf{persistence} of entropy from one step to the next. A value close to 1 means slow convergence (large $\tau_d = 1/\mu_d$), while a value close to 0 means rapid decorrelation.
    \item $c_{t,d}$ (\texttt{intercept} in the code): captures the pull toward the target $\theta_d$. When $\varphi \approx 1$ (weak mean-reversion), $c \approx \mu_d \theta_d \approx 0$ (for $\theta_d = 0$).
    \item $\sigma^\varepsilon_{t,d}$ (\texttt{sigma} in the code): the innovation noise, reflecting the volatility of the reverse diffusion process.
\end{itemize}

\subsubsection{Estimation Procedure}

\paragraph{Population-level AR(1) fitting.}
Rather than fitting a separate AR(1) per sample (which would have only $T$ observations), we pool samples within each class to estimate the parameters. For each class $\ell \in \{\text{correct}, \text{hallucination}\}$, step $t \in \{1,\dots,T\}$, and token position $d \in \{1,\dots,D\}$, we fit:
\begin{equation}
    X^{(i)}_{t,d} = \varphi^{(\ell)}_{t,d}\, X^{(i)}_{t-1,d} + c^{(\ell)}_{t,d} + \text{residual}, \qquad i \in \mathcal{I}_\ell
\end{equation}
via ordinary least squares (equivalently, \texttt{linregress}), where $\mathcal{I}_\ell$ is the set of sample indices belonging to class $\ell$. This yields:
\begin{align}
    \hat\varphi^{(\ell)}_{t,d} &= \frac{\mathrm{Cov}_i(X^{(i)}_{t-1,d},\, X^{(i)}_{t,d})}{\mathrm{Var}_i(X^{(i)}_{t-1,d})}, \\
    \hat c^{(\ell)}_{t,d} &= \bar X_{t,d} - \hat\varphi^{(\ell)}_{t,d}\,\bar X_{t-1,d}, \\
    \hat\sigma^{(\ell)}_{t,d} &= \mathrm{Std}_i\bigl(X^{(i)}_{t,d} - \hat\varphi^{(\ell)}_{t,d}\,X^{(i)}_{t-1,d} - \hat c^{(\ell)}_{t,d}\bigr).
\end{align}

\paragraph{Padding exclusion.}
When padding tokens are present (tokens where the model always proposes \texttt{pad\_token\_id} at all masked steps), they are excluded from the regression: only samples where token $d$ is \emph{not} padding contribute to the fit for position $d$. This prevents the degenerate behaviour of always-padding tokens from contaminating the AR(1) estimates.

\paragraph{Data split.}
The dataset is split 50/50 (stratified by label):
\begin{enumerate}
    \item \textbf{AR(1) fitting half}: used to estimate $(\hat\varphi^{(\ell)}, \hat c^{(\ell)}, \hat\sigma^{(\ell)})$ for each class.
    \item \textbf{Evaluation half}: used to compute per-sample features and train the downstream classifier.
\end{enumerate}

\subsubsection{AR(1)-Derived Features}

Given the two fitted models (correct and hallucination), we compute the following features for each test sample $i$:

\paragraph{Mean reconstruction prediction.}
For each model $\ell$, the one-step-ahead prediction on masked (non-padding) tokens:
\begin{equation}
    \hat X^{(i,\ell)}_{t,d} = \hat\varphi^{(\ell)}_{t,d}\,X^{(i)}_{t-1,d} + \hat c^{(\ell)}_{t,d}
\end{equation}
The \textbf{mean prediction} feature averages the predicted values across all masked non-padding tokens and steps:
\begin{equation}
    \mathrm{MeanPred}^{(i,\ell)} = \frac{\sum_{t,d} m^{(i)}_{t,d}\,(1-p^{(i)}_d)\,\hat X^{(i,\ell)}_{t,d}}{\sum_{t,d} m^{(i)}_{t,d}\,(1-p^{(i)}_d)}
\end{equation}
where $m^{(i)}_{t,d}$ is the mask and $p^{(i)}_d$ the padding indicator.

\paragraph{Variance reconstruction prediction.}
At each step, the inter-token variance of the predictions among masked tokens:
\begin{equation}
    \mathrm{VarPred}^{(i,\ell)}(t) = \mathrm{Var}_{d: m_{t,d}=1,\, p_d=0}\!\bigl(\hat X^{(i,\ell)}_{t,d}\bigr)
\end{equation}
averaged over steps:
\begin{equation}
    \mathrm{VarPred}^{(i,\ell)} = \frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}} \mathrm{VarPred}^{(i,\ell)}(t).
\end{equation}

\paragraph{Feature vector.}
The four AR(1) features used in the downstream logistic regression are:
\begin{equation}
    \mathbf{f}^{(i)}_{\mathrm{AR1}} = \bigl(\mathrm{MeanPred}^{(i,\text{corr})},\;\mathrm{MeanPred}^{(i,\text{hall})},\;\mathrm{VarPred}^{(i,\text{corr})},\;\mathrm{VarPred}^{(i,\text{hall})}\bigr)^\top.
\end{equation}

\paragraph{Intuition.}
\begin{itemize}
    \item A correct sample's entropy trajectory should be well-predicted by the correct model (low $\varphi^{\text{corr}}$, fast convergence) and poorly predicted by the hallucination model.
    \item A hallucinated sample has slower convergence ($\varphi$ closer to 1, higher persistence), so the hallucination model's predictions match better.
    \item The \emph{difference} $\mathrm{MeanPred}^{(i,\text{corr})} - \mathrm{MeanPred}^{(i,\text{hall})}$ acts as a discriminative score: positive values indicate hallucination-like dynamics.
\end{itemize}

\subsubsection{Connection to the OU Parameters}

The fitted AR(1) coefficients can be mapped back to continuous-time OU parameters:
\begin{align}
    \hat\mu_d(t) &= -\ln\hat\varphi_{t,d}, \\
    \hat\theta_d(t) &= \frac{\hat c_{t,d}}{1 - \hat\varphi_{t,d}}, \\
    \hat\sigma_d(t) &= \hat\sigma^\varepsilon_{t,d}\sqrt{\frac{2\hat\mu_d(t)}{1 - \hat\varphi_{t,d}^2}}.
\end{align}
This allows physical interpretation:
\begin{center}
\begin{tabular}{llll}
\toprule
\textbf{AR(1) param.} & \textbf{OU param.} & \textbf{Correct} & \textbf{Hallucinated} \\
\midrule
$\varphi_{t,d}$ & $e^{-\mu_d}$ & small (fast decay) & close to 1 (slow decay) \\
$c_{t,d}$ & $\theta_d(1-e^{-\mu_d})$ & pulls toward 0 & weak pull \\
$\sigma^\varepsilon_{t,d}$ & innovation noise & lower (stable) & higher (volatile) \\
\bottomrule
\end{tabular}
\end{center}

The mean-reversion rate $\mu_d$ is the inverse of the characteristic timescale $\tau_d$ from the Markovian model: $\mu_d = 1/\tau_d$. Thus the AR(1) framework provides a \emph{data-driven}, \emph{per-token, per-step} estimate of the convergence dynamics, whereas the Markovian model gives the theoretical envelope.

\subsubsection{Time-Inhomogeneous Calibration}

In practice, the OU parameters vary significantly with the diffusion step $t$, reflecting the non-stationary nature of the reverse diffusion process:
\begin{itemize}
    \item \textbf{Early steps} ($t$ small): $\sigma_d(t)$ is large (high stochasticity as many tokens are still masked), $\mu_d(t)$ is small (weak mean-reversion, entropy stays high).
    \item \textbf{Middle steps}: $\mu_d(t)$ increases as the scheduler unmasks tokens more aggressively, causing the mean entropy to decay.
    \item \textbf{Late steps} ($t$ large): $\mu_d(t)$ is very large for already-converged tokens (instantaneous convergence), while ``hard'' tokens still have moderate $\mu_d$.
\end{itemize}
This time-inhomogeneity is captured automatically by fitting separate $(\varphi_{t,d}, c_{t,d}, \sigma_{t,d})$ at each step $t$, rather than assuming constant coefficients.
```
