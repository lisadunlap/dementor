# 📁 Hierarchical Math Disguise - Directory Structure

## ✅ **Corrected Directory Organization**

The Microsoft Phi-4-mini-instruct results have been moved to the proper location to match the hierarchical math disguise structure.

### **Current Structure:**
```
disguising/scores/hierarchical_math_disguise/
├── source_meta-llama/
│   ├── Meta-Llama-3-8B-Instruct/
│   │   └── target_gpt/
│   │       └── gpt-5/
│   │           ├── heuristic_table.csv
│   │           └── heuristic_table_target_source.csv
│   └── target_gpt-5/
│       ├── comparison_results.csv
│       ├── heuristic_table.csv
│       └── heuristic_table_target_source.csv
└── source_microsoft/
    └── Phi-4-mini-instruct/
        └── target_gpt-5/
            ├── comparison_results.csv
            ├── heuristic_table.csv
            └── heuristic_table_target_source.csv
```

### **Files Moved:**
- ✅ `comparison_results.csv` (3.9MB) - Detailed LLM scoring results
- ✅ `heuristic_table.csv` (588B) - Style similarity heuristics  
- ✅ `heuristic_table_target_source.csv` (598B) - Target vs source heuristics

### **Key Results Files:**
1. **`comparison_results.csv`** - Contains semantic and stylistic scores for all 500 responses
2. **`heuristic_table.csv`** - Contains detailed style matching percentages
3. **`heuristic_table_target_source.csv`** - Contains target vs source style comparisons

### **Updated Scripts:**
- ✅ `aggregate_results.py` - Updated to use correct paths
- ✅ `detailed_aggregation.py` - Updated to use correct paths
- ✅ All aggregation scripts now work with the corrected structure

### **Verification:**
- ✅ All files are in the correct `target_gpt-5` directory
- ✅ Directory structure matches Meta Llama organization
- ✅ Aggregation scripts work correctly with new paths
- ✅ Results are properly organized for analysis

---

*Directory structure corrected on: September 5, 2025*  
*All Microsoft Phi-4-mini-instruct results now properly organized*
