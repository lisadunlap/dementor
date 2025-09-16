"""
Composite method: Contrastive rules + AL-selected examples.
"""
from typing import List, Dict, Any
import numpy as np
import pandas as pd

try:
    from .base import MethodBase
    from .contrastive import ContrastiveSystemPrompting
except ImportError:
    try:
        from scripts.methods.base import MethodBase
        from scripts.methods.contrastive import ContrastiveSystemPrompting
    except ImportError:
        from base import MethodBase
        from contrastive import ContrastiveSystemPrompting

try:
    from scripts.utils import get_token_count
except ImportError:
    from utils import get_token_count


class ContrastiveWithALExamples(MethodBase):
    """Contrastive rules + actively selected examples (AL/clustering/random)."""
    def __init__(self, model: str, disguise_as: str, disguise_df: pd.DataFrame = None,
                 source_df: pd.DataFrame = None, num_examples: int = 5,
                 al_kwargs: Dict[str, Any] = None, selector: str = 'al'):
        super().__init__(model, disguise_as)
        if disguise_df is None or source_df is None:
            raise ValueError("contrastive_with_al_examples requires disguise_df and source_df")
        self.disguise_df = disguise_df.copy()
        self.source_df = source_df.copy()
        self.num_examples = num_examples
        self.al_kwargs = al_kwargs or {}
        self.selector = selector

        self._contrastive = ContrastiveSystemPrompting(
            model=model,
            disguise_as=disguise_as,
            disguise_df=self.disguise_df,
            source_df=self.source_df,
            num_examples=num_examples,
        )

        self._al_selector = None
        if self.selector == 'al':
            try:
                from ..utils.active_learning_selector import ActiveLearningSelector
            except Exception:
                try:
                    from scripts.methods.utils.active_learning_selector import ActiveLearningSelector
                except ImportError:
                    from utils.active_learning_selector import ActiveLearningSelector
            self._al_selector = ActiveLearningSelector(
                target_responses_df=self.disguise_df.rename(columns={"target_response": "model_response"}),
                num_examples=self.num_examples,
                **self.al_kwargs,
            )

    def forward(self, prompt: str) -> List[Dict[str, str]]:
        contrastive_msgs = self._contrastive.forward(prompt)
        if len(contrastive_msgs) and contrastive_msgs[0]["role"] == "system":
            base_system = contrastive_msgs[0]["content"]
        else:
            base_system = f"You are {self.disguise_as}."

        if self.selector == 'al' and self._al_selector is not None:
            selected = self._al_selector.select_examples()
        elif self.selector == 'clustering':
            try:
                from ..utils.extract_styles import extract_style_features
            except Exception:
                try:
                    from scripts.methods.utils.extract_styles import extract_style_features
                except ImportError:
                    from utils.extract_styles import extract_style_features
            from sklearn.cluster import KMeans
            vecs = []
            for _, row in self.disguise_df.iterrows():
                txt = row.get('target_response', '')
                vecs.append(extract_style_features(txt))
            X = np.vstack(vecs) if len(vecs) else np.zeros((0, 1))
            n_clusters = min(self.num_examples, X.shape[0]) if X.shape[0] else 0
            if n_clusters <= 0:
                selected = self.disguise_df.head(0)
            else:
                km = KMeans(n_init=5, n_clusters=n_clusters)
                labels = km.fit_predict(X)
                centers = km.cluster_centers_
                chosen_idx = []
                for c in range(n_clusters):
                    idxs = np.where(labels == c)[0]
                    if len(idxs) == 0:
                        continue
                    diffs = np.linalg.norm(X[idxs] - centers[c], axis=1)
                    best = idxs[np.argmin(diffs)]
                    chosen_idx.append(best)
                selected = self.disguise_df.iloc[chosen_idx]
        else:
            selected = self.disguise_df.sample(n=min(self.num_examples, len(self.disguise_df)))

        examples_block = "\n\nReference examples in this style:\n"
        for _, row in selected.iterrows():
            q = row.get('prompt', '')
            a = row.get('model_response', row.get('target_response', ''))
            if isinstance(a, str) and len(a) > 800:
                a = a[:800] + "...(truncated)"
            examples_block += f"Q: {str(q)[:200]}...\nA: {a}\n\n"

        system_prompt = f"{base_system}{examples_block}\nRespond to the following prompt in this style."

        if "gemma" in self.model.lower():
            formatted = f"""<start_of_turn>user
{system_prompt}

{prompt}<end_of_turn>
<start_of_turn>model
"""
            return [{"role": "user", "content": formatted}]
        else:
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
