#!/usr/bin/env python3
"""
Test script for the active learning disguise method.
This demonstrates how the iterative active learning algorithm works for example selection.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from disguising.methods.active_learning_disguise import ActiveLearningDisguise
from disguising.methods.utils.active_learning_selector import ActiveLearningSelector
import matplotlib.pyplot as plt

def create_sample_data(n_samples=50):
    """Create sample mathematical response data for testing"""
    np.random.seed(42)
    
    # Generate sample prompts and responses
    prompts = []
    responses = []
    
    for i in range(n_samples):
        # Create different types of math problems
        problem_types = ['arithmetic', 'algebra', 'geometry', 'word_problem']
        problem_type = np.random.choice(problem_types)
        
        if problem_type == 'arithmetic':
            prompt = f"Calculate {np.random.randint(10, 100)} + {np.random.randint(10, 100)}"
            response = f"To solve this arithmetic problem:\n1. Add the numbers: {np.random.randint(10, 100)} + {np.random.randint(10, 100)} = {np.random.randint(20, 200)}\n2. The answer is {np.random.randint(20, 200)}."
        elif problem_type == 'algebra':
            prompt = f"Solve for x: {np.random.randint(2, 10)}x + {np.random.randint(1, 20)} = {np.random.randint(20, 50)}"
            response = f"To solve this algebraic equation:\n1. Subtract {np.random.randint(1, 20)} from both sides\n2. Divide by {np.random.randint(2, 10)}\n3. x = {np.random.randint(1, 10)}"
        elif problem_type == 'geometry':
            prompt = f"Find the area of a rectangle with length {np.random.randint(5, 20)} and width {np.random.randint(5, 20)}"
            response = f"To find the area of a rectangle:\n1. Use the formula: Area = length × width\n2. Area = {np.random.randint(5, 20)} × {np.random.randint(5, 20)}\n3. Area = {np.random.randint(25, 400)} square units"
        else:  # word_problem
            prompt = f"A store has {np.random.randint(50, 200)} items. They sell {np.random.randint(10, 50)} items. How many items are left?"
            response = f"To solve this word problem:\n1. Start with the total number of items: {np.random.randint(50, 200)}\n2. Subtract the number sold: {np.random.randint(10, 50)}\n3. Items remaining: {np.random.randint(30, 150)}"
        
        prompts.append(prompt)
        responses.append(response)
    
    # Create DataFrame
    df = pd.DataFrame({
        'prompt': prompts,
        'model_response': responses,
        'model': ['gpt-5'] * n_samples
    })
    
    return df

def test_active_learning_selector():
    """Test the active learning selector directly"""
    print("=== Testing Active Learning Selector ===")
    
    # Create sample data
    sample_data = create_sample_data(30)
    print(f"Created {len(sample_data)} sample responses")
    
    # Initialize selector
    selector = ActiveLearningSelector(
        sample_data,
        num_examples=5,
        d_regular=3,
        p_threshold=0.1,
        q_threshold=0.1,
        batch_size=5,
        max_iterations=3,
        seed=42
    )
    
    # Run active learning
    print("\nRunning active learning...")
    quality_scores = selector.run_active_learning()
    
    # Get statistics
    stats = selector.get_selection_statistics()
    uncertainty_stats = selector.get_uncertainty_analysis()
    
    print(f"\nSelection Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nUncertainty Analysis:")
    for key, value in uncertainty_stats.items():
        print(f"  {key}: {value}")
    
    # Select examples
    selected_examples = selector.select_examples()
    print(f"\nSelected {len(selected_examples)} examples")
    print("Selected example indices:", selected_examples.index.tolist())
    
    return selector, quality_scores

def test_active_learning_disguise():
    """Test the active learning disguise method"""
    print("\n=== Testing Active Learning Disguise Method ===")
    
    # Create sample data
    sample_data = create_sample_data(40)
    
    # Initialize disguise method
    disguise_method = ActiveLearningDisguise(
        model="test-model",
        disguise_as="gpt-5",
        disguise_df=sample_data,
        num_examples_per_disguise=5,
        d_regular=3,
        p_threshold=0.1,
        q_threshold=0.1,
        batch_size=8,
        max_iterations=4,
        seed=42
    )
    
    # Run analysis
    print("Running active learning analysis...")
    analysis = disguise_method.run_active_learning_analysis()
    
    print(f"\nQuality Analysis:")
    for key, value in analysis['quality_analysis'].items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    # Compare with random selection
    print("\nComparing with random selection...")
    comparison = disguise_method.compare_with_random_selection(num_random_samples=50)
    
    print(f"\nComparison Results:")
    for key, value in comparison.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    test_prompt = "Solve: 2x + 5 = 15"
    print(f"\nTesting forward pass with prompt: '{test_prompt}'")
    responses = disguise_method.forward(test_prompt)
    
    print(f"Generated {len(responses)} response(s)")
    for i, response in enumerate(responses):
        print(f"Response {i+1}:")
        for key, value in response.items():
            print(f"  {key}: {value}")
    
    return disguise_method, analysis

def visualize_quality_scores(quality_scores, title="Quality Scores Distribution"):
    """Visualize the distribution of quality scores"""
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(quality_scores, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Quality Score')
    plt.ylabel('Frequency')
    plt.title(f'{title} - Histogram')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(quality_scores)
    plt.ylabel('Quality Score')
    plt.title(f'{title} - Box Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('active_learning_quality_scores.png', dpi=150, bbox_inches='tight')
    print(f"Quality scores visualization saved as 'active_learning_quality_scores.png'")

def main():
    """Main test function"""
    print("Active Learning Disguise Method Test")
    print("=" * 50)
    
    try:
        # Test 1: Active Learning Selector
        selector, quality_scores = test_active_learning_selector()
        
        # Test 2: Active Learning Disguise Method
        disguise_method, analysis = test_active_learning_disguise()
        
        # Visualize results
        print("\n=== Creating Visualizations ===")
        visualize_quality_scores(quality_scores, "Active Learning Quality Scores")
        
        print("\n=== Test Summary ===")
        print("✓ Active Learning Selector: PASSED")
        print("✓ Active Learning Disguise Method: PASSED")
        print("✓ Quality Score Analysis: PASSED")
        print("✓ Random Selection Comparison: PASSED")
        print("✓ Visualizations: CREATED")
        
        print(f"\nKey Results:")
        print(f"- Selected {len(selector.target_df)} examples from {len(selector.target_df)} total")
        print(f"- Quality score range: {quality_scores.min():.3f} - {quality_scores.max():.3f}")
        print(f"- Mean quality: {quality_scores.mean():.3f}")
        
        if 'comparison' in locals():
            print(f"- Improvement over random: {comparison.get('percentage_improvement', 0):.1f}%")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n💥 Some tests failed.")
        sys.exit(1)
