import re
import numpy as np
from typing import Dict, List, Tuple

def has_step_numbering(response: str) -> bool:
    """Check if response uses numbered steps (e.g., 'Step 1:', '1.', etc.)"""
    step_patterns = [
        r'Step\s+\d+:',  # Step 1:, Step 2:, etc.
        r'^\d+\.',        # 1., 2., etc. at line start
        r'^\d+\)',        # 1), 2), etc. at line start
        r'Step\s+\d+',    # Step 1, Step 2, etc.
    ]
    return any(re.search(pattern, response, re.MULTILINE) for pattern in step_patterns)

def count_math_symbols(response: str) -> int:
    """Count mathematical symbols and notation in the response"""
    math_symbols = [
        '+', '-', '*', '/', '=', '<', '>', '≤', '≥', '≠',
        '×', '÷', '±', '√', '²', '³', '^', '(', ')', '[', ']',
        'π', '∞', '∑', '∫', '∂', '∆', '∇', '∈', '∉', '⊂', '⊃'
    ]
    count = 0
    for symbol in math_symbols:
        count += response.count(symbol)
    return count

def has_intermediate_calculations(response: str) -> bool:
    """Check if response shows intermediate calculation steps"""
    calc_patterns = [
        r'\d+\s*[+\-*/×÷]\s*\d+\s*=\s*\d+',  # 5 + 3 = 8
        r'=\s*\d+',                             # = 15
        r'→\s*\d+',                             # → 20
        r'=\s*[^=]+',                           # = 2 × 10
    ]
    return any(re.search(pattern, response) for pattern in calc_patterns)

def has_problem_analysis(response: str) -> bool:
    """Check if response starts with problem analysis/understanding"""
    analysis_patterns = [
        r'Step\s*1.*[Aa]nalysis',
        r'[Uu]nderstand.*problem',
        r'[Ff]irst.*[Ff]ind',
        r'[Ii]dentify.*[Gg]iven',
        r'[Gg]iven.*[Ii]nformation',
    ]
    return any(re.search(pattern, response) for pattern in analysis_patterns)

def has_verification_steps(response: str) -> bool:
    """Check if response includes verification or checking steps"""
    verification_patterns = [
        r'[Vv]erify',
        r'[Cc]heck',
        r'[Tt]est',
        r'[Pp]roof',
        r'[Cc]onfirm',
        r'[Vv]alidate',
    ]
    return any(re.search(pattern, response) for pattern in verification_patterns)

def extract_step_structure(response: str) -> Dict[str, any]:
    """Extract the complete step structure of a mathematical response"""
    lines = response.split('\n')
    steps = []
    current_step = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for step headers
        step_match = re.match(r'(Step\s+(\d+):?|^(\d+)[\.\)]:?)', line)
        if step_match:
            if current_step:
                steps.append(current_step)
            step_num = step_match.group(2) or step_match.group(3)
            current_step = {
                'number': int(step_num),
                'header': line,
                'content': [],
                'calculations': [],
                'has_math': False
            }
        elif current_step:
            current_step['content'].append(line)
            # Check if line contains math
            if re.search(r'[\d+\-*/×÷=]', line):
                current_step['has_math'] = True
            # Check if line contains calculations
            if re.search(r'\d+\s*[+\-*/×÷]\s*\d+\s*=\s*\d+', line):
                current_step['calculations'].append(line)
    
    if current_step:
        steps.append(current_step)
    
    return {
        'total_steps': len(steps),
        'steps': steps,
        'has_numbered_steps': len(steps) > 0,
        'avg_step_length': np.mean([len(step['content']) for step in steps]) if steps else 0
    }

def extract_math_notation(response: str) -> Dict[str, any]:
    """Extract mathematical notation patterns"""
    notation_patterns = {
        'fractions': len(re.findall(r'\d+/\d+', response)),
        'decimals': len(re.findall(r'\d+\.\d+', response)),
        'percentages': len(re.findall(r'\d+%', response)),
        'equations': len(re.findall(r'[^=]*=.*', response)),
        'inequalities': len(re.findall(r'[^<>]*[<>≤≥≠].*', response)),
        'parentheses': len(re.findall(r'[\(\)]', response)),
        'brackets': len(re.findall(r'[\[\]]', response)),
    }
    
    return notation_patterns

def extract_reasoning_patterns(response: str) -> Dict[str, any]:
    """Extract logical reasoning patterns"""
    reasoning_patterns = {
        'conditional_logic': len(re.findall(r'[Ii]f.*[Tt]hen|[Ii]f.*[Ss]o', response)),
        'causal_connections': len(re.findall(r'[Bb]ecause|[Ss]ince|[Aa]s|[Tt]herefore', response)),
        'comparisons': len(re.findall(r'[Mm]ore than|[Ll]ess than|[Ee]qual to|[Ss]ame as', response)),
        'definitions': len(re.findall(r'[Mm]eans|[Dd]efine|[Ii]s.*[Ww]hen', response)),
        'examples': len(re.findall(r'[Ff]or example|[Ee]xample|[Ss]uch as', response)),
    }
    
    return reasoning_patterns

def extract_problem_breakdown(response: str) -> Dict[str, any]:
    """Extract how the problem is broken down"""
    breakdown_patterns = {
        'given_info': len(re.findall(r'[Gg]iven|[Gg]iven.*[Ii]nformation', response)),
        'what_to_find': len(re.findall(r'[Ff]ind|[Ww]hat.*[Ii]s|[Hh]ow.*[Mm]any', response)),
        'assumptions': len(re.findall(r'[Aa]ssume|[Aa]ssuming|[Pp]resume', response)),
        'constraints': len(re.findall(r'[Cc]onstraint|[Ll]imit|[Mm]aximum|[Mm]inimum', response)),
        'units': len(re.findall(r'\d+\s*(miles?|hours?|dollars?|pounds?|meters?|seconds?)', response)),
    }
    
    return breakdown_patterns

def extract_comprehensive_math_features(response: str) -> Dict[str, any]:
    """Extract all mathematical style features from a response"""
    features = {}
    
    # Basic math features
    features['has_step_numbering'] = has_step_numbering(response)
    features['math_symbol_count'] = count_math_symbols(response)
    features['has_intermediate_calculations'] = has_intermediate_calculations(response)
    features['has_problem_analysis'] = has_problem_analysis(response)
    features['has_verification_steps'] = has_verification_steps(response)
    
    # Step structure analysis
    step_structure = extract_step_structure(response)
    features.update(step_structure)
    
    # Mathematical notation
    notation = extract_math_notation(response)
    features.update(notation)
    
    # Reasoning patterns
    reasoning = extract_reasoning_patterns(response)
    features.update(reasoning)
    
    # Problem breakdown
    breakdown = extract_problem_breakdown(response)
    features.update(breakdown)
    
    # Text-based features
    features['total_length'] = len(response)
    features['line_count'] = len(response.split('\n'))
    features['word_count'] = len(response.split())
    
    return features

def compare_math_style_similarity(response1: str, response2: str) -> float:
    """Compare the mathematical style similarity between two responses"""
    features1 = extract_comprehensive_math_features(response1)
    features2 = extract_comprehensive_math_features(response2)
    
    # Convert to numerical arrays for comparison
    feature_names = sorted(features1.keys())
    values1 = [features1[name] for name in feature_names]
    values2 = [features2[name] for name in feature_names]
    
    # Normalize values
    max_vals = [max(v1, v2) for v1, v2 in zip(values1, values2)]
    max_vals = [max(1, v) for v in max_vals]  # Avoid division by zero
    
    normalized1 = [v1 / max_val for v1, max_val in zip(values1, max_vals)]
    normalized2 = [v2 / max_val for v2, max_val in zip(values2, max_vals)]
    
    # Calculate cosine similarity
    dot_product = sum(n1 * n2 for n1, n2 in zip(normalized1, normalized2))
    norm1 = np.sqrt(sum(n1 * n1 for n1 in normalized1))
    norm2 = np.sqrt(sum(n2 * n2 for n2 in normalized2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return similarity
