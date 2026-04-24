from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import os
import plotly.graph_objects as go



def getTxt(Sequences, save_path, proposed_sequence = True):
    directory = f"StudyAutoRegressionBehavior/{save_path}"

    if not os.path.exists(directory):
        os.makedirs(directory) 

        
    for j in range(len(Sequences)):
        file_name = f"StudyAutoRegressionBehavior/{save_path}/proposed_text EX - {j}.txt" if proposed_sequence else f"StudyAutoRegressionBehavior/{save_path}/generated_text EX - {j}.txt"
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(f'==================== MESSAGE {j} ====================\n')
            for r in range(len(Sequences[j])):
                tokenss = Sequences[j][r]
                f.write(f'{r} . {tokenss}\n')
            f.write('\n\n')
            f.write(f'Final sentence: {" ".join(Sequences[j][-1])}\n')


def getAdaptativeMaskforPlot(Masks):
    res_masks = []
    for j in range(len(Masks)):
        M = np.array(Masks[j])
 
        indices_rows = np.argmax(M == 0, axis=0)    
        has_zero = np.any(M == 0, axis=0) 
        res = np.zeros_like(M)
        cols = np.where(has_zero)[0]
        rows = indices_rows[has_zero]
        res[rows, cols] = 1
        res_masks.append(res)
    return res_masks


def _get_remask_data_for_sample(remasking_masks, sample_idx, num_samples):
    remasking_array = np.array(remasking_masks)
    if remasking_array.ndim == 3 and remasking_array.shape[1] == num_samples:
        return remasking_array[:, sample_idx, :].astype(bool)
    return np.array(remasking_masks[sample_idx]).astype(bool)


_SPECIAL_TOKEN_REPR = {
    ' ':  '[SPACE]',
    '\n': '[NEWLINE]',
    '\t': '[TAB]',
    '\r': '[CARRIAGE_RETURN]',
    '':   '[EMPTY]',
    '<|mdm_mask|>': '[MDM_MASK]',
}

def _make_visible_tokens(tokens_matrix):
    """Replace invisible/whitespace tokens with visible symbols for hover display."""
    def _vis(tok):
        if tok in _SPECIAL_TOKEN_REPR:
            return _SPECIAL_TOKEN_REPR[tok]
        stripped = tok.strip()
        if stripped == '' and tok != '':
            return f'[{repr(tok)}]'
        return tok
    return [[_vis(tok) for tok in row] for row in tokens_matrix]



def plotLogProbs(LogProbsList, title_list, save_path, blcok_size, sequences, masks, remasking_masks=None):
    masks_adaptative = getAdaptativeMaskforPlot(masks)
    for j in range(len(LogProbsList)):
        masks_adaptative_j = masks_adaptative[j]
        token_matrix = _make_visible_tokens(sequences[j])
        generated_text = token_matrix[-1]
        data = np.array(LogProbsList[j])
        n_rows, n_cols = data.shape
        plt.figure(figsize=(25, 15))
        ax = plt.gca()
        ax.set_facecolor((1, 0, 0, 0.3))        
        sns.heatmap(data, mask=masks_adaptative_j, cmap='viridis', vmin=0, vmax=1, cbar=True, ax=ax)

        if remasking_masks is not None:
            remasking_array = np.array(remasking_masks)
            if remasking_array.ndim == 3 and remasking_array.shape[1] == len(LogProbsList):
                remask_data = remasking_array[:, j, :].astype(bool)
            else:
                remask_data = np.array(remasking_masks[j]).astype(bool)
            remask_overlay = np.zeros((n_rows, n_cols), dtype=float)
            copy_rows = min(n_rows, remask_data.shape[0])
            copy_cols = min(n_cols, remask_data.shape[1])
            remask_overlay[:copy_rows, :copy_cols] = remask_data[:copy_rows, :copy_cols].astype(float)
            remask_overlay = np.ma.masked_where(remask_overlay == 0, remask_overlay)
            ax.imshow(
                remask_overlay,
                cmap=ListedColormap([(0.0, 0.0, 0.0, 1.0)]),
                interpolation='none',
                aspect='auto',
                origin='upper',
                extent=(0, n_cols, n_rows, 0),
                zorder=3,
            )

        plt.ylabel('Diffusion iterations')
        plt.xlabel('tokens index')
        plt.title(f'Convergence of LogProbs: Diff. iterations VS Index ({title_list[j]})')        
        n_blocks = n_cols // blcok_size
        for i in range(n_blocks + 1):
            plt.axvline(blcok_size * i, linewidth=4, color='red', linestyle='--', 
                        label='Block size' if i == 0 else "")
        
        ax.set_xticks(np.arange(len(generated_text)) + 0.5)
        ax.set_xticklabels(generated_text, rotation=90, fontsize=8)

        y_pos = np.arange(n_rows)
        y_labs = np.arange(1, n_rows + 1)
        ax.set_yticks(y_pos[::3] + 0.5)
        ax.set_yticklabels(y_labs[::3])

        ax.set_xticks(np.arange(len(generated_text) + 1), minor=True)
        ax.set_yticks(np.arange(n_rows + 1), minor=True)
        
        ax.tick_params(which='major', bottom=True, left=True, length=5, color='black')
        ax.tick_params(which='minor', bottom=False, left=False)
        
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.grid(which='major', visible=False) 
        
        plt.xlim(0, n_cols)
        plt.ylim(n_rows, 0)
        mask_handles = [
            Patch(facecolor=(1, 0, 0, 0.3), edgecolor='black', label='Beginning of unmasking'),
            Patch(facecolor=(0.0, 0.0, 0.0, 1.0), edgecolor='black', label='Remasking'),
            Line2D([0], [0], color='red', linestyle='--', linewidth=4, label='Block size')
        ] if remasking_masks is not None else [
            Patch(facecolor=(1, 0, 0, 0.3), edgecolor='black', label='Beginning of unmasking'),
            Line2D([0], [0], color='red', linestyle='--', linewidth=4, label='Block size')
        ]
        
        plt.legend(handles=mask_handles)
        plt.tight_layout()
        
        plt.savefig(f'StudyAutoRegressionBehavior/{save_path}/logprobs_convergence_{title_list[j]}.png')
        plt.show()



def plotUnmaskLogProbs(UnmaskLogProbsList, title_list, save_path, blcok_size, sequences, masks, remasking_masks=None):
    masks_adaptative = getAdaptativeMaskforPlot(masks)
    for j in range(len(UnmaskLogProbsList)):
        masks_adaptative_j = masks_adaptative[j]
        token_matrix = _make_visible_tokens(sequences[j])
        generated_text = token_matrix[-1]
        data = np.array(UnmaskLogProbsList[j])
        n_rows, n_cols = data.shape
        
        plt.figure(figsize=(25, 15))
        ax = plt.gca()
        ax.set_facecolor((1, 0, 0, 0.3))        
        sns.heatmap(data, mask=masks_adaptative_j, cmap='viridis', vmin=0, vmax=1, cbar=True, ax=ax)

        if remasking_masks is not None:
            remasking_array = np.array(remasking_masks)
            if remasking_array.ndim == 3 and remasking_array.shape[1] == len(UnmaskLogProbsList):
                remask_data = remasking_array[:, j, :].astype(bool)
            else:
                remask_data = np.array(remasking_masks[j]).astype(bool)
            remask_overlay = np.zeros((n_rows, n_cols), dtype=float)
            copy_rows = min(n_rows, remask_data.shape[0])
            copy_cols = min(n_cols, remask_data.shape[1])
            remask_overlay[:copy_rows, :copy_cols] = remask_data[:copy_rows, :copy_cols].astype(float)
            remask_overlay = np.ma.masked_where(remask_overlay == 0, remask_overlay)
            ax.imshow(
                remask_overlay,
                cmap=ListedColormap([(0.0, 0.0, 0.0, 1.0)]),
                interpolation='none',
                aspect='auto',
                origin='upper',
                extent=(0, n_cols, n_rows, 0),
                zorder=3,
            )

        plt.ylabel('Diffusion iterations')
        plt.xlabel('tokens index')
        plt.title(f'Convergence of UnmaskLogProbs: Diff. iterations VS Index ({title_list[j]})')
        
        n_blocks = n_cols // blcok_size
        for i in range(n_blocks + 1):
            plt.axvline(blcok_size * i, linewidth=4, color='red', linestyle='--', 
                        label='Block size' if i == 0 else "")
        
        ax.set_xticks(np.arange(len(generated_text)) + 0.5)
        ax.set_xticklabels(generated_text, rotation=90, fontsize=8)

        y_pos = np.arange(n_rows)
        y_labs = np.arange(1, n_rows + 1)
        ax.set_yticks(y_pos[::3] + 0.5)
        ax.set_yticklabels(y_labs[::3])
        
        ax.set_xticks(np.arange(len(generated_text) + 1), minor=True)
        ax.set_yticks(np.arange(n_rows + 1), minor=True)
        
        ax.tick_params(which='major', bottom=True, left=True, length=5, color='black')
        ax.tick_params(which='minor', bottom=False, left=False)
        
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.grid(which='major', visible=False) 
        
        plt.xlim(0, n_cols)
        plt.ylim(n_rows, 0)
        mask_handles = [
            Patch(facecolor=(1, 0, 0, 0.3), edgecolor='black', label='Beginning of unmasking'),
            Patch(facecolor=(0.0, 0.0, 0.0, 1.0), edgecolor='black', label='Remasking'),
            Line2D([0], [0], color='red', linestyle='--', linewidth=4, label='Block size')
        ] if remasking_masks is not None else [
            Patch(facecolor=(1, 0, 0, 0.3), edgecolor='black', label='Beginning of unmasking'),
            Line2D([0], [0], color='red', linestyle='--', linewidth=4, label='Block size')
        ]
        
        plt.legend(handles=mask_handles)
        plt.tight_layout()
        
        plt.savefig(f'StudyAutoRegressionBehavior/{save_path}/unmasklogprobs_convergence_{title_list[j]}.png')
        plt.show()
        

def plotMasks(Masks, title_list, save_path, blcok_size, sequences, remasking=False):
    for j in range(len(Masks)):
        token_matrix = _make_visible_tokens(sequences[j])
        generated_text = token_matrix[-1]
        data = np.array(Masks[j])
        n_rows, n_cols = data.shape
        
        plt.figure(figsize=(25, 15))
        
        ax = plt.gca()
        plt.imshow(data, vmin=0, vmax=1, cmap='gray', aspect='auto')
        
        plt.ylabel('Diffusion iterations')
        plt.xlabel('tokens index')
        plt.title(f'Evolution of masks across diffusion ({title_list[j]})') if not remasking else plt.title(f'Evolution of remasks across diffusion ({title_list[j]})')
        
        n_blocks = n_cols // blcok_size
        for i in range(n_blocks + 1):
            plt.axvline(blcok_size * i, linewidth=4, color='red', linestyle='--', 
                        label='Block size' if i == 0 else "")
        
        ax.set_xticks(np.arange(len(generated_text)))
        ax.set_xticklabels(generated_text, rotation=90, fontsize=8)

        y_pos = np.arange(n_rows)
        y_labs = np.arange(1, n_rows + 1)
        ax.set_yticks(y_pos[::3] + 0.5)
        ax.set_yticklabels(y_labs[::3])
        
        ax.set_xticks(np.arange(-0.5, len(generated_text), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        
        ax.tick_params(which='major', bottom=True, left=True, length=5, color='black')
        ax.tick_params(which='minor', bottom=False, left=False)
        
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.grid(which='major', visible=False) 
        
        plt.xlim(-0.5, n_cols - 0.5)
        plt.ylim(n_rows - 0.5, -0.5)
        
        mask_handles = [
            Patch(facecolor='black', edgecolor='black', label='mask False'),
            Patch(facecolor='white', edgecolor='black', label='mask True'),
        ]
        plt.legend(handles=mask_handles + ax.get_legend_handles_labels()[0])
        
        plt.tight_layout()
        plt.savefig(f'StudyAutoRegressionBehavior/{save_path}/mask_evolution_LLaDa_{title_list[j]}.png') if not remasking else plt.savefig(f'StudyAutoRegressionBehavior/{save_path}/remask_evolution_LLaDa_{title_list[j]}.png')
        plt.show()

def plotChanges(changes, title_list, save_path, blcok_size, sequences):
    for j in range(len(changes)):
        token_matrix = _make_visible_tokens(sequences[j])
        generated_text = token_matrix[-1]
        data = np.array(changes[j])
        n_rows, n_cols = data.shape
        
        # --- PREMIER PLOT : Binary Changes ---
        plt.figure(figsize=(25, 15))
        ax1 = plt.gca()
        plt.imshow(data, vmin=0, vmax=1, cmap='gray', aspect='auto')
        
        plt.ylabel('Diffusion iterations')
        plt.xlabel('tokens index')
        plt.title(f'Changes of tokenss between two steps of diffusion ({title_list[j]})')
        
        n_blocks = n_cols // blcok_size
        for i in range(n_blocks + 1):
            plt.axvline(blcok_size * i, linewidth=4, color='red', linestyle='--', 
                        label='Block size' if i == 0 else "")
        
        ax1.set_xticks(np.arange(len(generated_text)))
        ax1.set_xticklabels(generated_text, rotation=90, fontsize=8)

        y_pos = np.arange(n_rows)
        y_labs = np.arange(1, n_rows + 1)
        ax1.set_yticks(y_pos[::3] + 0.5)
        ax1.set_yticklabels(y_labs[::3])
        
        ax1.set_xticks(np.arange(-0.5, len(generated_text), 1), minor=True)
        ax1.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        
        ax1.tick_params(which='major', bottom=True, left=True, length=5, color='black')
        ax1.tick_params(which='minor', bottom=False, left=False)
        
        ax1.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax1.grid(which='major', visible=False) 
        
        plt.xlim(-0.5, n_cols - 0.5)
        plt.ylim(n_rows - 0.5, -0.5)
        
        mask_handles = [
            Patch(facecolor='white', edgecolor='black', label='change True'),
            Patch(facecolor='black', edgecolor='black', label='change False'),
        ]
        plt.legend(handles=mask_handles + ax1.get_legend_handles_labels()[0])
        
        plt.tight_layout()
        plt.savefig(f'StudyAutoRegressionBehavior/{save_path}/logprobs_changes_LLaDa_{title_list[j]}.png')
        plt.show()

        # --- DEUXIÈME PLOT : Cumulative Changes ---
        plt.figure(figsize=(25, 15))
        res_changes_cumulative = np.cumsum(data, axis=0)
        ax2 = sns.heatmap(res_changes_cumulative, cmap='viridis', vmin=0, vmax=np.max(res_changes_cumulative))
        
        plt.ylabel('Diffusion iterations')
        plt.xlabel('tokens index')
        plt.title(f'Cumulative changes in logprobs across diffusion ({title_list[j]})')
        
        for i in range(n_blocks + 1):
            plt.axvline(blcok_size * i, linewidth=4, color='red', linestyle='--', 
                        label='Block size' if i == 0 else "")
        
        ax2.set_xticks(np.arange(len(generated_text)) + 0.5)
        ax2.set_xticklabels(generated_text, rotation=90, fontsize=8)

        y_pos = np.arange(n_rows)
        y_labs = np.arange(1, n_rows + 1)
        ax2.set_yticks(y_pos[::3] + 0.5)
        ax2.set_yticklabels(y_labs[::3])
        
        ax2.set_xticks(np.arange(len(generated_text) + 1), minor=True)
        ax2.set_yticks(np.arange(n_rows + 1), minor=True)
        
        ax2.tick_params(which='major', bottom=True, left=True, length=5, color='black')
        ax2.tick_params(which='minor', bottom=False, left=False)
        
        ax2.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax2.grid(which='major', visible=False) 
        
        plt.xlim(0, n_cols)
        plt.ylim(n_rows, 0)
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'StudyAutoRegressionBehavior/{save_path}/Changes_{title_list[j]}.png')
        plt.show()



def plotLevenshtein(levenshteins, title_list, save_path, blcok_size, sequences):
    for j in range(len(levenshteins)):
        token_matrix = _make_visible_tokens(sequences[j])
        generated_text = token_matrix[-1]
        data = np.array(levenshteins[j])
        n_rows, n_cols = data.shape
        
        plt.figure(figsize=(25, 15))
        ax = sns.heatmap(data, cmap='viridis', vmin=0, vmax=np.max(data), cbar=True)
        
        plt.ylabel('Diffusion iterations')
        plt.xlabel('tokens index order')
        plt.title(f'Levenshtein distance: Diffusion VS tokens index order ({title_list[j]})')
        
        n_blocks = n_cols // blcok_size
        for i in range(n_blocks + 1):
            plt.axvline(blcok_size * i, linewidth=4, color='red', linestyle='--', 
                        label='Block size' if i == 0 else "")
        
        ax.set_xticks(np.arange(len(generated_text)) + 0.5)
        ax.set_xticklabels(generated_text, rotation=90, fontsize=8)
        
        y_pos = np.arange(n_rows)
        y_labs = np.arange(1, n_rows + 1)
        ax.set_yticks(y_pos[::3] + 0.5)
        ax.set_yticklabels(y_labs[::3])
        
        ax.set_xticks(np.arange(len(generated_text) + 1), minor=True)
        ax.set_yticks(np.arange(n_rows + 1), minor=True)
        
        ax.tick_params(which='major', bottom=True, left=True, length=5, color='black')
        ax.tick_params(which='minor', bottom=False, left=False)
        
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.grid(which='major', visible=False) 
        
        plt.xlim(0, n_cols)
        plt.ylim(n_rows, 0)
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'StudyAutoRegressionBehavior/{save_path}/heatmap_levenshtein_distance_LLaDa_{title_list[j]}.png')
        plt.show()

def plotAttentionMask(attention_masks, title_list, save_path):
    plt.figure(figsize=(25, 15))
    positions = np.arange(len(title_list))
    plt.yticks(ticks=positions, labels=title_list) 
    plt.imshow(attention_masks, cmap='gray', aspect='auto')
    plt.title('Attention masks for each example')
    mask_handles = [
        Patch(facecolor='black', label='mask False'),
        Patch(facecolor='white', edgecolor='black', label='mask True'),
    ]    
    plt.legend(handles=mask_handles, loc='upper right')    
    plt.xlabel('tokens index order')
    plt.ylabel('Examples')    
    plt.grid()    
    plt.savefig(f'StudyAutoRegressionBehavior/{save_path}/attention_masks.png', bbox_inches='tight')
    plt.show()


def plotH(Hs, title_list, save_path, blcok_size, sequences, masks, remasking_masks=None):
    adaptative_masks = getAdaptativeMaskforPlot(masks)
    for j in range(len(Hs)):
        token_matrix = _make_visible_tokens(sequences[j])
        generated_text = token_matrix[-1]
        data = np.array(Hs[j])
        n_rows, n_cols = data.shape
        
        plt.figure(figsize=(25, 15))
        ax = plt.gca()
        ax.set_facecolor((1, 0, 0, 0.3))  
        ax = sns.heatmap(data, mask=adaptative_masks[j], cmap='viridis', vmin=-1, vmax=1, cbar=True)
        
        plt.ylabel('Diffusion iterations')
        plt.xlabel('tokens index order')
        plt.title(f'H: Diffusion VS tokens index order ({title_list[j]})')
        
        n_blocks = n_cols // blcok_size
        for i in range(n_blocks + 1):
            plt.axvline(blcok_size * i, linewidth=4, color='red', linestyle='--', 
                        label='Block size' if i == 0 else "")
        
        if remasking_masks is not None:
            remasking_array = np.array(remasking_masks)
            if remasking_array.ndim == 3 and remasking_array.shape[1] == len(Hs):
                remask_data = remasking_array[:, j, :].astype(bool)
            else:
                remask_data = np.array(remasking_masks[j]).astype(bool)
            remask_overlay = np.zeros((n_rows, n_cols), dtype=float)
            copy_rows = min(n_rows, remask_data.shape[0])
            copy_cols = min(n_cols, remask_data.shape[1])
            remask_overlay[:copy_rows, :copy_cols] = remask_data[:copy_rows, :copy_cols].astype(float)
            remask_overlay = np.ma.masked_where(remask_overlay == 0, remask_overlay)
            ax.imshow(
                remask_overlay,
                cmap=ListedColormap([(0.0, 0.0, 0.0, 1.0)]),
                interpolation='none',
                aspect='auto',
                origin='upper',
                extent=(0, n_cols, n_rows, 0),
                zorder=3,
            )
        
        ax.set_xticks(np.arange(len(generated_text)) + 0.5)
        ax.set_xticklabels(generated_text, rotation=90, fontsize=8)
        
        y_pos = np.arange(n_rows)
        y_labs = np.arange(1, n_rows + 1)
        ax.set_yticks(y_pos[::3] + 0.5)
        ax.set_yticklabels(y_labs[::3])
        
        ax.set_xticks(np.arange(len(generated_text) + 1), minor=True)
        ax.set_yticks(np.arange(n_rows + 1), minor=True)
        
        ax.tick_params(which='major', bottom=True, left=True, length=5, color='black')
        ax.tick_params(which='minor', bottom=False, left=False)
        
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.grid(which='major', visible=False) 
        mask_handles = [
            Patch(facecolor=(1, 0, 0, 0.3), edgecolor='black', label='Beginning of unmasking'),
            Patch(facecolor=(0.0, 0.0, 0.0, 1.0), edgecolor='black', label='Remasking'),
            Line2D([0], [0], color='red', linestyle='--', linewidth=4, label='Block size')
        ] if remasking_masks is not None else [
            Patch(facecolor=(1, 0, 0, 0.3), edgecolor='black', label='Beginning of unmasking'),
            Line2D([0], [0], color='red', linestyle='--', linewidth=4, label='Block size')
        ]
        plt.xlim(0, n_cols)
        plt.ylim(n_rows, 0)
        
        plt.legend(handles=mask_handles)
        plt.tight_layout()
        plt.savefig(f'StudyAutoRegressionBehavior/{save_path}/H_{title_list[j]}.png')
        plt.show()
    
def plotNumTransferredTokens(NumTransferredTokens, title_list, save_path):
    plt.figure(figsize=(25, 15))
    for j in range(len(NumTransferredTokens)):
        sns.heatmap(NumTransferredTokens[j], cmap='viridis', cbar=True, annot=True,)
        plt.xlabel('Diffusion Iteration')
        plt.ylabel('Number of Transferred Tokens')
        plt.title('Number of Transferred Tokens Across Diffusion Iterations')
        plt.legend()
        plt.grid()
        plt.savefig(f'StudyAutoRegressionBehavior/{save_path}/num_transferred_tokens_{title_list[j]}.png')
        plt.show()

def PlotlyLogProbs(logprobs, title_list, save_path, block_size, sequences, masks, remasking_masks=None):
    masks_adaptative = getAdaptativeMaskforPlot(masks)
    for j in range(len(logprobs)):
        data = np.array(logprobs[j])
        n_rows, n_cols = data.shape
        tokens_matrix = _make_visible_tokens(sequences[j])
        data[masks_adaptative[j] == 1] = np.nan
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=list(range(n_cols)),
            y=list(range(1, n_rows + 1)),
            showscale=True,
            customdata=tokens_matrix,
            hovertemplate=(
                    "<b>Token: %{customdata}</b><br>" +
                    "Itération: %{y}<br>" +
                    "Index: %{x}<br>" +
                    "Valeur: %{z:.4f}" +
                    "<extra></extra>"
                        ),
            colorscale='Viridis',
            zmin=0,
            zmax=1,
            colorbar=dict(
            title="LogProbs",     
                ),
        ))

        if remasking_masks is not None:
            remask_data = _get_remask_data_for_sample(remasking_masks, j, len(logprobs))
            remask_overlay = np.full((n_rows, n_cols), np.nan)
            copy_rows = min(n_rows, remask_data.shape[0])
            copy_cols = min(n_cols, remask_data.shape[1])
            remask_overlay[:copy_rows, :copy_cols] = remask_data[:copy_rows, :copy_cols].astype(float)
            fig.add_trace(go.Heatmap(
                z=remask_overlay,
                x=list(range(n_cols)),
                y=list(range(1, n_rows + 1)),
                showscale=False,
                hoverinfo='skip',
                colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,1)']],
                zmin=0,
                zmax=1,
            ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=12, color='salmon', symbol='square'),
            name='Beginning of unmasking',
            showlegend=True
        ))

        if remasking_masks is not None:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=12, color='black', symbol='square'),
                name='Remasking',
                showlegend=True
            ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Block size'
        ))

        for i in range((n_cols // block_size) + 1):
            fig.add_vline(x=i * block_size - 0.5, line_width=2, line_dash="dash", line_color="red")
        fig.update_layout(
            title=f"Convergence: {title_list[j]}",
            template="plotly_white",  
            plot_bgcolor='salmon',    
            paper_bgcolor='#F8F9F9',  
            
            xaxis=dict(
                            title="tokens index",
                            dtick=1,
                            showticklabels=True,
                            ticks='outside',
                            ticklen=6,
                            tickwidth=1,
                            tickcolor='black',
                            showline=True,
                            linecolor='black',
                            showgrid=False
                        ),            
            yaxis=dict(title="Diffusion Iterations", autorange='reversed', dtick=3, showgrid=False),
            
            showlegend=True,
            legend=dict(
                orientation="h",    
                yanchor="bottom",
                y=1.05,               
                xanchor="right",
                x=1
            ),
            width=1400,
            height=850,
            margin=dict(t=150)        
        )
        filename = f"StudyAutoRegressionBehavior/{save_path}/logprobs_convergence_{title_list[j]}.html"
        fig.write_html(filename)



def PlotlyUnmaskLogProbs(UnmaskLogProbsList, title_list, save_path, block_size, sequences, masks, remasking_masks=None):
    masks_adaptative = getAdaptativeMaskforPlot(masks)
    for j in range(len(UnmaskLogProbsList)):
        data = np.array(UnmaskLogProbsList[j])
        data[masks_adaptative[j] == 1] = np.nan
        n_rows, n_cols = data.shape
        tokens_matrix = _make_visible_tokens(sequences[j])
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=list(range(n_cols)),
            y=list(range(1, n_rows + 1)),
            showscale=True,
            customdata=tokens_matrix,
            hovertemplate=(
                    "<b>Token: %{customdata}</b><br>" +
                    "Itération: %{y}<br>" +
                    "Index: %{x}<br>" +
                    "Valeur: %{z:.4f}" +
                    "<extra></extra>"
                        ),
            colorscale='Viridis',
            zmin=0,
            zmax=1,
            colorbar=dict(
            title="UnmaskLogProbs",     
                ),
        ))

        if remasking_masks is not None:
            remask_data = _get_remask_data_for_sample(remasking_masks, j, len(UnmaskLogProbsList))
            remask_overlay = np.full((n_rows, n_cols), np.nan)
            copy_rows = min(n_rows, remask_data.shape[0])
            copy_cols = min(n_cols, remask_data.shape[1])
            remask_overlay[:copy_rows, :copy_cols] = remask_data[:copy_rows, :copy_cols].astype(float)
            fig.add_trace(go.Heatmap(
                z=remask_overlay,
                x=list(range(n_cols)),
                y=list(range(1, n_rows + 1)),
                showscale=False,
                hoverinfo='skip',
                colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,1)']],
                zmin=0,
                zmax=1,
            ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=12, color='salmon', symbol='square'),
            name='Beginning of unmasking',
            showlegend=True
        ))

        if remasking_masks is not None:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=12, color='black', symbol='square'),
                name='Remasking',
                showlegend=True
            ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Block size'
        ))

        for i in range((n_cols // block_size) + 1):
            fig.add_vline(x=i * block_size - 0.5, line_width=2, line_dash="dash", line_color="red")
        fig.update_layout(
            title=f"Convergence of UnmaskLogProbs: Diff. iterations VS Index VS unmasked token ({title_list[j]})",
            template="plotly_white",  
            plot_bgcolor='salmon',    
            paper_bgcolor='#F8F9F9',  
            xaxis=dict(
                title="tokens index",
                dtick=1,
                showticklabels=True,
                ticks='outside',
                ticklen=6,
                tickwidth=1,
                tickcolor='black',
                showline=True,
                linecolor='black',
                showgrid=False
            ),
            yaxis=dict(
                title="Diffusion iterations",
                autorange='reversed', 
                dtick=3,
                showgrid=False 
            ),
            width=1400,
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",    
                yanchor="bottom",
                y=1.05,               
                xanchor="right",
                x=1
            ),
            margin=dict(t=150)        

        )
        filename = f"StudyAutoRegressionBehavior/{save_path}/unmasklogprobs_convergence_{title_list[j]}.html"
        fig.write_html(filename)


def PlotlyChanges(changesList, title_list, save_path, block_size, sequences, remasking_masks=None):
    for j in range(len(changesList)):
        generated_text = sequences[j][-1]
        data = np.array(changesList[j]).astype(int)
        n_rows, n_cols = data.shape
        tokens_matrix = _make_visible_tokens(sequences[j])

        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=list(range(n_cols)),
            y=list(range(1, n_rows + 1)),
            customdata=tokens_matrix,
            hovertemplate=(
                    "<b>Token: %{customdata}</b><br>" +
                    "Itération: %{y}<br>" +
                    "Index: %{x}<br>" +
                    "Valeur: %{z}<extra></extra>"
            ),
            colorscale=[[0, 'black'], [1, 'white']],
            zmin=0,
            zmax=1,
            showscale=False
        ))

        if remasking_masks is not None:
            remask_data = _get_remask_data_for_sample(remasking_masks, j, len(changesList))
            remask_overlay = np.full((n_rows, n_cols), np.nan)
            copy_rows = min(n_rows, remask_data.shape[0])
            copy_cols = min(n_cols, remask_data.shape[1])
            remask_overlay[:copy_rows, :copy_cols] = remask_data[:copy_rows, :copy_cols].astype(float)
            fig.add_trace(go.Heatmap(
                z=remask_overlay,
                x=list(range(n_cols)),
                y=list(range(1, n_rows + 1)),
                showscale=False,
                hoverinfo='skip',
                colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,1)']],
                zmin=0,
                zmax=1,
            ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='white', symbol='square', line=dict(width=1, color='black')),
            legendgroup='True',
            showlegend=True,
            name='Change True (1)'
        ))

        if remasking_masks is not None:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color='black', symbol='square', line=dict(width=1, color='white')),
                showlegend=True,
                name='Remasking'
            ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='black', symbol='square'),
            legendgroup='False',
            showlegend=True,
            name='Change False (0)'
        ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Block size'
        ))
        for i in range((n_cols // block_size) + 1):
            fig.add_vline(x=i * block_size - 0.5, line_width=2, line_dash="dash", line_color="red")

        fig.update_layout(
            title=f"Changes of tokens (proposed tokens) ({title_list[j]})",
            template="plotly_white",  
            paper_bgcolor='#F8F9F9',
            plot_bgcolor='black', 
            
            xaxis=dict(
                title="tokens index",
                dtick=1,
                showticklabels=True,
                ticks='outside',
                ticklen=6,
                tickwidth=1,
                tickcolor='black',
                showline=True,
                linecolor='black',
                showgrid=False
            ),
            yaxis=dict(
                title="Diffusion iterations",
                autorange='reversed', 
                dtick=3,
                showticklabels=True,
                ticks='outside',
                ticklen=6,
                tickwidth=1,
                tickcolor='black',
                showline=True,
                linecolor='black',
                showgrid=False
            ),
            width=1400,
            height=800,
            showlegend=True
        )
        filename = f"StudyAutoRegressionBehavior/{save_path}/Changes_{title_list[j]}.html"
        fig.write_html(filename)

def PlotlyH(Hs, title_list, save_path, block_size, sequences, masks, remasking_masks=None):
    adaptative_masks = getAdaptativeMaskforPlot(masks)
    for j in range(len(Hs)):
        data = np.array(Hs[j])
        n_rows, n_cols = data.shape
        tokens_matrix = _make_visible_tokens(sequences[j])
        data[adaptative_masks[j] == 1] = np.nan
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=list(range(n_cols)),
            y=list(range(1, n_rows + 1)),
            showscale=True,
            customdata=tokens_matrix,
            hovertemplate=(
                    "<b>Token: %{customdata}</b><br>" +
                    "Itération: %{y}<br>" +
                    "Index: %{x}<br>" +
                    "Valeur: %{z:.4f}" +
                    "<extra></extra>"
                        ),
            colorscale='Viridis',
            zmin=-1,
            zmax=1,
            colorbar=dict(
            title="H",     
                ),
        ))

        if remasking_masks is not None:
            remask_data = _get_remask_data_for_sample(remasking_masks, j, len(Hs))
            remask_overlay = np.full((n_rows, n_cols), np.nan)
            copy_rows = min(n_rows, remask_data.shape[0])
            copy_cols = min(n_cols, remask_data.shape[1])
            remask_overlay[:copy_rows, :copy_cols] = remask_data[:copy_rows, :copy_cols].astype(float)
            fig.add_trace(go.Heatmap(
                z=remask_overlay,
                x=list(range(n_cols)),
                y=list(range(1, n_rows + 1)),
                showscale=False,
                hoverinfo='skip',
                colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,1)']],
                zmin=0,
                zmax=1,
            ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=12, color='salmon', symbol='square'),
            name='Beginning of unmasking',
            showlegend=True
        ))

        if remasking_masks is not None:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=12, color='black', symbol='square'),   

                name='Remasking',
                showlegend=True
            ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Block size'
        ))
        for i in range((n_cols // block_size) + 1):
            fig.add_vline(x=i * block_size - 0.5, line_width=2, line_dash="dash", line_color="red") 
        fig.update_layout(
            title=f"H: Diffusion VS tokens index order ({title_list[j]})",
            template="plotly_white",  
            plot_bgcolor='salmon',    
            paper_bgcolor='#F8F9F9',  
            xaxis=dict(
                title="tokens index",
                dtick=1,
                showticklabels=True,
                ticks='outside',
                ticklen=6,
                tickwidth=1,
                tickcolor='black',
                showline=True,
                linecolor='black',
                showgrid=False
            ),
            yaxis=dict(
                title="Diffusion iterations",
                autorange='reversed', 
                dtick=3,
                showgrid=False 
            ),
            width=1400,
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",    
                yanchor="bottom",
                y=1.05,               
                xanchor="right",
                x=1
            ),
            margin=dict(t=150)        

        )
        filename = f"StudyAutoRegressionBehavior/{save_path}/H_{title_list[j]}.html"
        fig.write_html(filename)

