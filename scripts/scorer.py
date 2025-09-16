"""
Consolidated scoring utilities for evaluating model responses.
Combines functionality from llm_scorer.py and llm_scorer2.py into a clean interface.
"""
import pandas as pd
import re
import os
from typing import Optional, Tuple, Dict, List
import argparse
import sys
import logging
import json

# Load environment variables from .env if available (for OPENAI_API_KEY, etc.)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Optional imports with fallbacks
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    print("Warning: VLLM not available. LLM judge scoring will be disabled.")
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

try:
    # Prefer module under scripts/methods/utils
    from .methods.utils.stylistic_analysis import compute_heuristics
except Exception:
    try:
        # Fallback to scripts/stylistic_analysis.py
        from .stylistic_analysis import compute_heuristics
    except Exception:
        print("Warning: stylistic_analysis not available. Using fallback heuristics.")
        def compute_heuristics(responses1, responses2):
            """Fallback heuristics function that matches stylistic_analysis signature."""
            # Simple fallback: compare word counts
            data = []
            for r1, r2 in zip(responses1, responses2):
                word_count_match = abs(len(r1.split()) - len(r2.split())) < 10  # Within 10 words
                data.append({"word_count_match": word_count_match})
            return pd.DataFrame(data)

try:
    from .utils import get_token_count
except Exception:
    print("Warning: utils not available. Using fallback token counter.")
    def get_token_count(text):
        return len(text.split())


class ModelScorer:
    """Unified interface for scoring model responses."""
    
    def __init__(self, judge_model: str = "openai/gpt-4.1-mini", 
                 temperature: float = 0.0, max_tokens: int = 512):
        """
        Initialize scorer with judge model.
        
        Args:
            judge_model: Model to use for scoring (default: Phi-4-mini)
            temperature: Temperature for judge model
            max_tokens: Max tokens for judge responses
        """
        self.judge_model = judge_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.sampling_params = None
        if VLLM_AVAILABLE:
            try:
                self.sampling_params = SamplingParams(
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except Exception:
                self.sampling_params = None
        
    def _judge_backend(self) -> str:
        jm = (self.judge_model or "").lower()
        if jm.startswith("vllm:"):
            return "vllm"
        if jm.startswith("hf:") or jm.startswith("huggingface:"):
            return "hf"
        return "litellm"

    def _vllm_generate(self, prompt_text: str) -> str:
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not installed; cannot use vLLM judge backend")
        try:
            if self.llm is None:
                model_id = self.judge_model.split(":", 1)[1] if self.judge_model.lower().startswith("vllm:") else self.judge_model
                self.llm = LLM(model=model_id, gpu_memory_utilization=0.8)
                if self.sampling_params is None:
                    self.sampling_params = SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens)
            outputs = self.llm.generate([prompt_text], self.sampling_params)
            return outputs[0].outputs[0].text
        except Exception as e:
            logging.error(f"vLLM judge error: {e}")
            raise

    def _hf_generate(self, prompt_text: str) -> str:
        try:
            from transformers import pipeline
        except ImportError as e:
            raise RuntimeError("Transformers not installed. pip install transformers accelerate") from e
        model_id = self.judge_model.split(":", 1)[1]
        pipe = pipeline("text-generation", model=model_id, device_map="auto")
        out = pipe(prompt_text, max_new_tokens=self.max_tokens, do_sample=(self.temperature > 0), temperature=max(self.temperature, 1e-6))
        text = out[0]['generated_text']
        return text[len(prompt_text):].strip()

    def _litellm_generate(self, messages: List[dict]) -> str:
        try:
            from litellm import completion
            import litellm
            # Enable caching for API calls (not for vLLM/local servers)
            if not hasattr(litellm, 'cache') or litellm.cache is None:
                litellm.cache = litellm.Cache()
        except ImportError as e:
            raise RuntimeError("LiteLLM not installed. pip install litellm") from e
        resp = completion(model=self.judge_model, messages=messages, max_tokens=self.max_tokens, temperature=self.temperature)
        return resp["choices"][0]["message"]["content"]
    
    def _clean_thinking_output(self, output: str) -> str:
        """Remove <think> tags from model output."""
        pattern = r'<think>.*?</think>'
        cleaned = re.sub(pattern, '', output, flags=re.DOTALL)
        return re.sub(r'\n\s*\n', '\n\n', cleaned).strip()
    
    def score_similarity(self, response_a: str, response_b: str, 
                        prompt: Optional[str] = None) -> Dict[str, float]:
        """
        Score semantic and stylistic similarity between two responses.
        
        Args:
            response_a: First response
            response_b: Second response  
            prompt: Original prompt (optional, for context)
            
        Returns:
            Dict with semantic_score, stylistic_score, and explanations
        """
        backend = self._judge_backend()
        comparison_prompt = self._build_comparison_prompt(response_a, response_b, prompt)
        try:
            if backend == 'vllm':
                raw_output = self._vllm_generate(comparison_prompt)
            elif backend == 'hf':
                raw_output = self._hf_generate(comparison_prompt)
            else:  # LiteLLM
                system_msg = "You are an expert evaluator of AI model responses."
                messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": comparison_prompt}]
                raw_output = self._litellm_generate(messages)
            cleaned_output = self._clean_thinking_output(raw_output)
            scores = self._parse_scores(cleaned_output)
            scores['raw_explanation'] = cleaned_output
            return scores
        except Exception as e:
            logging.error(f"Error scoring responses: {e}")
            return {
                'semantic_score': 0.0,
                'stylistic_score': 0.0, 
                'raw_explanation': f"Error: {str(e)}"
            }
    
    def _build_comparison_prompt(self, response_a: str, response_b: str, 
                               prompt: Optional[str] = None) -> str:
        """Build comparison prompt for judge model."""
        system_prompt = """You are an expert evaluator of AI model responses. Your task is to rate the similarity between two responses in two key dimensions:

1. Semantic Meaning (1-4 points) - this ONLY considers the meaning of the response, not the style, formatting, or tone:
   - 4: Responses convey nearly identical meaning and information, with only trivial differences in wording or phrasing. If the format of the responses are different but the core meaning is the same, the score should be 4.
   - 3: Responses share the same core intent or function, even if one includes minor additional elements (e.g., both are greetings, both answer the same question, both provide the same instruction, etc.).
   - 2: Responses have some overlap in meaning, with significant differences in what is being communicated.
   - 1: Responses are unrelated in meaning or intent.

2. Stylistic Similarity (1-4 points) - this ONLY considers the style, formatting, and tone of the response, not the content:
   - 4: Response style is identical (formatting, tone, formality, etc.)
   - 3: Response style is similar but not identical (e.g., both are informal, both use bullet points, etc.)
   - 2: Response style has some elements in common (e.g., both are informal or conversational), but also significant differences in expressiveness, complexity, or structure
   - 1: Response style is completely different (e.g., one is formal, one is slang; one is a list, one is a paragraph)

For each comparison, provide:
1. The semantic score (1-4)
2. The stylistic score (1-4)
3. A brief explanation of the score breakdown"""

        if prompt:
            context = f"Original prompt: {prompt}\n\n"
        else:
            context = ""
            
        comparison_text = f"""{context}Response A:
{response_a}

Response B:
{response_b}

Please evaluate these responses and provide your scores and explanation."""

        return f"{system_prompt}\n\n{comparison_text}"
    
    def _parse_scores(self, output: str) -> Dict[str, float]:
        """Parse semantic and stylistic scores from judge output."""
        semantic_score = 0.0
        stylistic_score = 0.0
        
        # Look for various score patterns
        semantic_patterns = [
            r'[Ss]emantic[:\s]+(\d(?:\.\d)?)',
            r'[Ss]emantic[^:]*[:\s]+(\d(?:\.\d)?)', 
            r'(\d(?:\.\d)?)[^0-9]*semantic',
        ]
        
        stylistic_patterns = [
            r'[Ss]tylistic[:\s]+(\d(?:\.\d)?)',
            r'[Ss]tylistic[^:]*[:\s]+(\d(?:\.\d)?)',
            r'(\d(?:\.\d)?)[^0-9]*stylistic',
        ]
        
        for pattern in semantic_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                semantic_score = float(match.group(1))
                break
                
        for pattern in stylistic_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                stylistic_score = float(match.group(1))
                break
        
        return {
            'semantic_score': semantic_score,
            'stylistic_score': stylistic_score
        }
    
    def score_batch(self, df: pd.DataFrame, response_col_a: str = 'target_response',
                   response_col_b: str = 'model_response', 
                   prompt_col: str = 'prompt') -> pd.DataFrame:
        """
        Score a batch of response pairs.
        
        Args:
            df: DataFrame with response pairs
            response_col_a: Column name for first responses
            response_col_b: Column name for second responses
            prompt_col: Column name for prompts
            
        Returns:
            DataFrame with added scoring columns
        """
        results = []
        
        for idx, row in df.iterrows():
            scores = self.score_similarity(
                response_a=row[response_col_a],
                response_b=row[response_col_b],
                prompt=row.get(prompt_col, None)
            )
            results.append(scores)
            
            if idx % 10 == 0:
                logging.info(f"Scored {idx+1}/{len(df)} response pairs")
        
        # Add results to dataframe
        result_df = df.copy()
        for key in ['semantic_score', 'stylistic_score', 'raw_explanation']:
            result_df[key] = [r[key] for r in results]
            
        return result_df
    
    def compute_heuristic_scores(self, df: pd.DataFrame, 
                               response_col_a: str = 'target_response',
                               response_col_b: str = 'model_response') -> pd.DataFrame:
        """
        Compute heuristic similarity scores (fast, no LLM).
        
        Args:
            df: DataFrame with response pairs
            response_col_a: Column name for first responses  
            response_col_b: Column name for second responses
            
        Returns:
            DataFrame with heuristic scores added
        """
        result_df = df.copy()
        
        # Compute heuristics between the two response columns
        heuristics = compute_heuristics(df[response_col_a].tolist(), df[response_col_b].tolist())
        
        # The compute_heuristics function returns a DataFrame of boolean matches
        # Convert to match scores (1 for match, 0 for no match) and average across all features
        result_df['heuristic_match_score'] = heuristics.mean(axis=1)
        
        # Add individual heuristic comparisons as columns
        for col in heuristics.columns:
            result_df[f'heuristic_{col}'] = heuristics[col]
            
        return result_df
    
    def _calculate_heuristic_match(self, heuristics_a: pd.DataFrame, 
                                 heuristics_b: pd.DataFrame) -> pd.Series:
        """Calculate overall heuristic match score between response pairs."""
        # Simple weighted average of individual feature matches
        match_scores = []
        
        for idx in range(len(heuristics_a)):
            row_a = heuristics_a.iloc[idx]
            row_b = heuristics_b.iloc[idx]
            
            # Calculate matches for each feature
            feature_matches = []
            
            # Length similarity
            len_a, len_b = row_a.get('response_length', 0), row_b.get('response_length', 0)
            if max(len_a, len_b) > 0:
                len_match = 1 - abs(len_a - len_b) / max(len_a, len_b)
            else:
                len_match = 1.0
            feature_matches.append(len_match)
            
            # Binary feature matches (headers, lists, etc.)
            binary_features = ['has_markdown_headers', 'has_code_blocks', 'has_bullet_points']
            for feature in binary_features:
                if feature in row_a and feature in row_b:
                    feature_matches.append(1.0 if row_a[feature] == row_b[feature] else 0.0)
            
            # Average match score
            overall_match = sum(feature_matches) / len(feature_matches) if feature_matches else 0.0
            match_scores.append(overall_match)
            
        return pd.Series(match_scores)


def score_model_comparison(input_file: str, output_file: Optional[str] = None,
                         heuristics_only: bool = False, 
                         judge_model: str = "gpt-4.1-mini") -> pd.DataFrame:
    """
    Convenience function for scoring model comparisons from file.
    
    Args:
        input_file: CSV file with response pairs
        output_file: Optional output file path
        heuristics_only: If True, only compute heuristic scores
        judge_model: Judge model to use for LLM scoring
        
    Returns:
        DataFrame with scores
    """
    # Load data with robust CSV parsing
    try:
        df = pd.read_csv(input_file)
    except pd.errors.ParserError:
        logging.warning("CSV parsing error detected, retrying with error handling...")
        try:
            df = pd.read_csv(input_file, on_bad_lines='skip', quoting=1)  # QUOTE_ALL
        except pd.errors.ParserError:
            logging.warning("Second parsing attempt failed, trying with minimal quoting...")
            df = pd.read_csv(input_file, on_bad_lines='skip', quoting=3, engine='python')  # QUOTE_NONE with python engine
        logging.info(f"Loaded {len(df)} rows after handling CSV issues")
    
    # Initialize scorer
    scorer = ModelScorer(judge_model=judge_model)
    
    # Compute scores
    if heuristics_only:
        logging.info("Computing heuristic scores only...")
        result_df = scorer.compute_heuristic_scores(df)
    else:
        logging.info("Computing LLM and heuristic scores...")
        result_df = scorer.score_batch(df)
        result_df = scorer.compute_heuristic_scores(result_df)
    
    # Save results and metrics
    if output_file:
        result_df.to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file}")

        # Write metrics CSV
        metrics = summarize_scores(result_df)
        try:
            base, ext = os.path.splitext(output_file)
            metrics_csv = f"{base}_metrics.csv"
            import csv
            with open(metrics_csv, 'w', newline='') as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(['metric', 'value'])
                for k, v in metrics.items():
                    writer.writerow([k, v])
            logging.info(f"Metrics saved to {metrics_csv}")
        except Exception as e:
            logging.warning(f"Could not write metrics.csv: {e}")
    
    return result_df


def summarize_scores(df: pd.DataFrame) -> Dict[str, float]:
    """Return simple aggregates for quick reporting.

    Expected columns if LLM scoring ran: 'semantic_score', 'stylistic_score'
    Expected columns if heuristics ran: 'heuristic_match_score'
    """
    out: Dict[str, float] = {}

    if 'semantic_score' in df.columns:
        valid_sem = df['semantic_score'].dropna()
        if len(valid_sem):
            out['semantic_score_mean'] = float(valid_sem.mean())
            out['semantic_score_std'] = float(valid_sem.std())
            out['semantic_score_count'] = int(valid_sem.shape[0])
    if 'stylistic_score' in df.columns:
        valid_style = df['stylistic_score'].dropna()
        if len(valid_style):
            out['stylistic_score_mean'] = float(valid_style.mean())
            out['stylistic_score_std'] = float(valid_style.std())
            out['stylistic_score_count'] = int(valid_style.shape[0])
    if 'heuristic_match_score' in df.columns:
        hm = df['heuristic_match_score'].dropna()
        if len(hm):
            out['heuristic_match_mean'] = float(hm.mean())
            out['heuristic_match_std'] = float(hm.std())
            out['heuristic_match_count'] = int(hm.shape[0])
    return out


def _cli():
    parser = argparse.ArgumentParser(description="Score model responses and print summary")
    parser.add_argument("input", help="CSV with response pairs")
    parser.add_argument("--output", help="Optional output CSV path")
    parser.add_argument("--heuristics-only", action="store_true", help="Compute heuristics only")
    parser.add_argument("--judge-model", default="openai/gpt-4.1-mini", help="Judge model id. Prefix with vllm: or hf: for local backends; provider ids (e.g., openai/gpt-4o-mini) use LiteLLM.")
    parser.add_argument("--openai-api-base", default=None, help="Override OPENAI_API_BASE for LiteLLM judge routing")
    parser.add_argument("--openai-api-key", default=None, help="Override OPENAI_API_KEY for LiteLLM judge routing")
    args = parser.parse_args()

    # Auto-generate output path if not provided
    if args.output is None:
        input_path = args.input
        base, ext = os.path.splitext(input_path)
        args.output = f"{base}_scores{ext}"

    # Ensure judge routing goes to the correct endpoint
    if args.openai_api_base:
        os.environ["OPENAI_API_BASE"] = args.openai_api_base
    else:
        # If using an OpenAI-provided model, default to the official OpenAI base
        jm = (args.judge_model or "").lower()
        if jm.startswith("openai/"):
            os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    df = score_model_comparison(
        input_file=args.input,
        output_file=args.output,
        heuristics_only=args.heuristics_only,
        judge_model=args.judge_model,
    )
    summary = summarize_scores(df)
    if summary:
        print("Summary:")
        for k, v in summary.items():
            print(f"- {k}: {v}")
    else:
        print("No scores to summarize.")


if __name__ == "__main__":
    _cli()
