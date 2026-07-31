```latex
\subsection{Semantic Token Clustering}
\label{subsec:semantic-clustering}

\subsubsection{Motivation}

Standard token-level entropy treats every token in the vocabulary as equally distinct. However, in practice, a model hesitating between semantically equivalent alternatives (e.g.\ a space and a newline, or ``colour'' and ``color'') does not indicate genuine uncertainty about the underlying meaning. Conversely, hesitation between semantically distant tokens (e.g.\ ``dog'' and ``equation'') is a strong signal of confusion and potential hallucination.

To capture this distinction, we introduce a \emph{semantic clustering} of the vocabulary and derive two complementary uncertainty measures that are invariant to benign surface-level hesitation.

\subsubsection{Clustering Procedure (STC)}

We follow the Semantic Token Clustering (STC) approach~\cite{stc_eacl2026} and adapt it to our diffusion language model (LLaDA-8B-Instruct / Dream-7B-Instruct).

\paragraph{Step 1: Embedding extraction.}
For each token $v$ in the vocabulary $\mathcal{V}$ (size $V$), we extract two representations:
\begin{itemize}
    \item The \textbf{input embedding} $\mathbf{e}^{\mathrm{in}}_v \in \mathbb{R}^{D}$ from the token embedding layer.
    \item The \textbf{output embedding} $\mathbf{e}^{\mathrm{out}}_v \in \mathbb{R}^{D}$ from the language modeling head (\texttt{lm\_head}).
\end{itemize}
The input embeddings capture morphological and syntactic similarity, while the output embeddings capture predictive semantic similarity. Their concatenation yields a richer unified representation:
\begin{equation}
    \mathbf{u}_v = [\mathbf{e}^{\mathrm{in}}_v \,\|\, \mathbf{e}^{\mathrm{out}}_v] \in \mathbb{R}^{2D}
\end{equation}

\paragraph{Step 2: Filtering.}
Two categories of tokens are excluded from the clustering and assigned individual singleton clusters:
\begin{itemize}
    \item \textbf{Stopwords} (NLTK English list): grammatical tokens without semantic content (``the'', ``is'', ``a'', etc.) that would otherwise contaminate clusters with unrelated content words.
    \item \textbf{Arabic numerals}: digits have close embeddings (e.g.\ ``42'' and ``43'') but carry very different factual meaning, so grouping them would incorrectly suppress meaningful uncertainty.
\end{itemize}

\paragraph{Step 3: Agglomerative Clustering.}
After L2-normalizing the unified embeddings (so that Euclidean distance equals cosine distance), we apply \textbf{Agglomerative Clustering} with average linkage and cosine metric, targeting $n = 3{,}000$ clusters (tuned empirically following~\cite{stc_eacl2026}).

Average linkage defines the inter-cluster distance as:
\begin{equation}
    d(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{a \in C_i} \sum_{b \in C_j} d_{\cos}(\mathbf{u}_a, \mathbf{u}_b)
\end{equation}
The algorithm starts from $V$ singleton clusters and iteratively merges the two closest clusters until $n$ clusters remain. This produces a deterministic, fixed mapping $\sigma: \mathcal{V} \to \{1, \dots, n + n_{\mathrm{singletons}}\}$ that is pre-computed offline once.

\paragraph{Validation.}
Clustering quality is assessed via the silhouette score (cosine metric) computed on a 50k-token subsample.

\subsubsection{Semantic Entropy}

Given the cluster assignment $\sigma$, at each diffusion step $t$ and token position $d$, the model produces a probability distribution $p_{t,d}(v)$ over the vocabulary. We aggregate probabilities by cluster:
\begin{equation}
    P_{t,d}(c) = \sum_{v:\, \sigma(v) = c} p_{t,d}(v), \qquad c \in \{1, \dots, C\}
\end{equation}
The \textbf{semantic entropy} is then the Shannon entropy of this coarse-grained distribution:
\begin{equation}
    H^{\mathrm{sem}}_{t,d} = -\sum_{c=1}^{C} P_{t,d}(c) \log P_{t,d}(c)
\end{equation}

\paragraph{Interpretation.}
If the model hesitates between tokens within the same cluster, $P_{t,d}$ is concentrated on a single cluster and $H^{\mathrm{sem}}_{t,d} \approx 0$ (benign hesitation). If it hesitates between tokens in different clusters, multiple $P_{t,d}(c)$ are large and $H^{\mathrm{sem}}_{t,d}$ is high (genuine semantic uncertainty).

\subsubsection{Semantic Dispersion}

As a complement that avoids discrete clustering entirely, we define the \textbf{semantic dispersion}---a continuous, hyperparameter-free measure of how spread the predictive distribution is in embedding space:
\begin{equation}
    D_{t,d} = 1 - \left\| \sum_{v=1}^{V} p_{t,d}(v) \, \hat{\mathbf{e}}_v \right\|^2 = 1 - \|\bar{\mathbf{e}}_{t,d}\|^2
\end{equation}
where $\hat{\mathbf{e}}_v = \mathbf{e}^{\mathrm{in}}_v / \|\mathbf{e}^{\mathrm{in}}_v\|$ are L2-normalized input embeddings and $\bar{\mathbf{e}}_{t,d}$ is the probability-weighted barycenter.

\paragraph{Properties.}
\begin{itemize}
    \item $D \in [0, 1]$: the measure is bounded.
    \item $D \approx 0$ when the distribution is concentrated on semantically close tokens ($\|\bar{\mathbf{e}}\| \approx 1$).
    \item $D$ is large when the distribution spreads mass over distant regions of the embedding space.
    \item No hyperparameters (no top-$K$, no distance threshold, no cluster count).
    \item Efficiently computable as a single matrix multiplication: $\bar{\mathbf{e}} = p \cdot \hat{E}$ followed by a squared norm.
\end{itemize}

\subsubsection{Derived Features}

Both $H^{\mathrm{sem}}_{t,d}$ and $D_{t,d}$ are recorded at every diffusion step $t$ and position $d$. We apply the same aggregation operators as for the baseline entropy (Section~\ref{subsec:baseline}):
\begin{itemize}
    \item \textbf{VarSemanticEntropy}: $\frac{1}{D}\sum_d \mathrm{Var}_t(H^{\mathrm{sem}}_{t,d})$ --- temporal variance per token, averaged.
    \item \textbf{VarSemanticEntropyAcrossTokens}: $\frac{1}{T}\sum_t \mathrm{Var}_d(H^{\mathrm{sem}}_{t,d})$ --- spatial variance per step, averaged.
    \item \textbf{VarSemanticEntropyMasked}: same as VarSemanticEntropy but restricted to masked steps.
    \item \textbf{VarSemanticEntropyMaskedAcrossTokens}: same as VarSemanticEntropyAcrossTokens but restricted to masked positions.
    \item \textbf{VarSemanticDispersion} / \textbf{VarSemanticDispersionAcrossTokens}: analogous variance features computed on $D_{t,d}$.
\end{itemize}

\subsubsection{Empirical Observations}

\paragraph{High correlation with baseline entropy.}
The semantic variants are strongly correlated with their standard-entropy counterparts (e.g.\ $\rho(\mathrm{VarEntropy}, \mathrm{VarSemanticEntropy}) = 0.99$, $\rho(\mathrm{VarMaskedEntropy}, \mathrm{VarSemanticEntropyMasked}) = 0.993$). This suggests that, in practice, most of the model's hesitation already occurs between semantically distinct tokens rather than surface-level variants.

\paragraph{Complementary signal.}
Despite the high correlation, feature selection (L1-regularised logistic regression) retains VarSemanticEntropyAcrossToken alongside standard features, indicating that the semantic grouping does contribute marginal discriminative information not fully captured by raw entropy.
```
