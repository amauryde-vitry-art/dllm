import numpy as np
import matplotlib.pyplot as plt


def smoothstep(x):
    """Interpolation C1 lisse entre 0 et 1."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

# ---------------------------------------------------------
# 1. Nouveaux paramètres calibrés (Forme ET Échelle)
# ---------------------------------------------------------
def mu_smooth(t):
    """Faible pour laisser V croître, fort à la fin pour la chute"""
    if t < 50:
        return 0.05
    else:
        # Augmentation forte pour provoquer la chute rapide
        return 0.05 + 0.09 * (t - 50)

def sigma_smooth(t):
    """Calibré pour: montée rapide initiale → plateau V≈2.4 → pic → chute"""
    if t < 8:
        # Phase 1 lissée: décroissance progressive de sigma pour éviter la cassure vers t≈5.
        # 0.58 -> 0.49 avec une transition cubique continue.
        u = smoothstep(t / 8.0)
        return 0.58 - 0.09 * u
    elif t < 40:
        # Phase 2: plateau à V≈2.4
        # Steady state: V_ss = σ²/(2μ) = 0.2401/0.10 = 2.40 ✓
        return 0.49
    elif t < 52:
        # Phase 3: montée vers le pic V≈5
        # σ monte de 0.49 → 0.71, peak V_ss = 0.71²/(2×0.05) = 5.04 ✓
        return 0.49 + 0.018 * (t - 40)
    else:
        # Phase 4: σ chute pour aider la descente
        return max(0.35, 0.71 - 0.07 * (t - 52))

def theta_smooth(t):
    """Cible de la moyenne"""
    return 0

# ---------------------------------------------------------
# 2. Configuration de la simulation
# ---------------------------------------------------------
T_max = 61
steps_data = 61
sub_steps = 50  # Précision de l'intégration numérique
N = steps_data * sub_steps
dt = T_max / N

t_sim = np.linspace(0, T_max, N)
V = np.zeros(N)
E = np.zeros(N)

# Conditions initiales lues sur vos vrais graphiques
V[0] = 1.45  # Var initiale
E[0] = 4.15  # Mean initiale

# ---------------------------------------------------------
# 3. Résolution des EDO
# ---------------------------------------------------------
for i in range(1, N):
    t_prev = t_sim[i-1]
    m = mu_smooth(t_prev)
    s = sigma_smooth(t_prev)
    th = theta_smooth(t_prev)
    
    # Évolution Moyenne
    dE = m * (th - E[i-1]) * dt
    E[i] = E[i-1] + dE
    
    # Évolution Variance
    dV = (-2 * m * V[i-1] + s**2) * dt
    V[i] = V[i-1] + dV
    if V[i] < 0: V[i] = 0

# ---------------------------------------------------------
# 4. Tracé des graphiques (Échelles verrouillées)
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150)

# Graphe de la Moyenne (Mean)
ax1.plot(t_sim, E, color='#e74c3c', linewidth=2.5, label=r'$\mathbb{E}[X_t]$ (Simulée)')
ax1.set_title("MeanMaskedEntropyAcrossTokens", fontsize=11, fontweight='bold')
ax1.set_xlabel("Generation Step")
ax1.set_ylabel("Moyenne")
ax1.set_xlim(-2, 65)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Graphe de la Variance (Var)
ax2.plot(t_sim, V, color='#c0392b', linewidth=2.5, linestyle='--', label=r'$\mathrm{Var}(X_t)$ (Simulée)')
ax2.set_title("VarMaskedEntropyAcrossTokens", fontsize=11, fontweight='bold')
ax2.set_xlabel("Generation Step")
ax2.set_ylabel("Variance")
ax2.set_xlim(-2, 65)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.suptitle("Simulation OU non-homogène (Échelles et Formes Validées)", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("PipelineTest/res/ProcessusOU.png", dpi=150, bbox_inches='tight')
plt.show()