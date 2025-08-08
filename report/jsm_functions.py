
def column_similarity(col1, col2):
    col1_embeddings = np.array(col1.tolist())
    col2_embeddings = np.array(col2.tolist())

    # Perform element-wise multiplication
    return np.sum(col1_embeddings * col2_embeddings, axis=1)    

import matplotlib.patheffects as pe
from adjustText import adjust_text
def pca_plot_revised2(embeddings, highlight_ids, target_concepts, ax, truncate_size=30, ylabel=True):
    pca = PCA(n_components=2)
    emb_matrix = np.vstack(embeddings.to_list())
    coords = pca.fit_transform(emb_matrix)
    var_ratio = pca.explained_variance_ratio_
    name_map = target_concepts.set_index('concept_id')['concept_name'].to_dict()
    ax.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.3, color='lightgrey', label='Other Concepts', zorder=1)
    mask = target_concepts['concept_id'].isin(highlight_ids).to_numpy()
    x_hl, y_hl = coords[mask, 0], coords[mask, 1]
    ax.scatter(x_hl, y_hl, s=80, alpha=0.9, color='#e41a1c', edgecolor='black', linewidth=0.5, label='Head and Neck Concepts', zorder=3)
    texts = []
    for xi, yi, cid in zip(x_hl, y_hl, target_concepts.loc[mask, 'concept_id']):
        txt = name_map[cid]
        if len(txt) > truncate_size:
            txt = txt[:truncate_size - 3] + '...'
        t = ax.text(
            xi, yi, txt,
            fontsize=11, fontweight='bold', color='#e41a1c', zorder=4,
            path_effects=[pe.withStroke(linewidth=2, foreground='white')]
        )
        texts.append(t)
    adjust_text(texts, ax=ax, expand_points=(1.2, 1.2), arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}% var)', fontsize=14)
    if ylabel:
        ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}% var)', fontsize=14)
    # ax.set_title('PCA Visualization with Highlighted Concepts', fontsize=16, pad=15)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, fontsize=12, loc='upper right')
    plt.tight_layout()
 