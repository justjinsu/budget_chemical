# K-PetChem Carbon Budget Toolkit - Troubleshooting Guide

## Common Import Errors and Solutions

### 1. "attempted relative import with no known parent package"

**Problem**: Streamlit is trying to run dashboard files directly, but they use relative imports.

**Solutions**:

#### Option A: Use the launcher script (RECOMMENDED)
```bash
cd kpetchem_budget/
python run_dashboard.py
```

#### Option B: Run from correct directory
```bash
cd kpetchem_budget/
streamlit run dashboard/app_upgraded.py
```

#### Option C: Set PYTHONPATH
```bash
cd kpetchem_budget/
export PYTHONPATH=".:$PYTHONPATH"
streamlit run dashboard/app_upgraded.py
```

### 2. "No module named 'data_layer'" or similar

**Problem**: Python can't find the required modules.

**Solutions**:

1. **Check your current directory**:
   ```bash
   pwd
   # Should show: .../kpetchem_budget
   ls
   # Should show: data_layer.py, parameter_space.py, pathway.py, etc.
   ```

2. **Ensure file structure is correct**:
   ```
   kpetchem_budget/
   ├── data_layer.py
   ├── parameter_space.py
   ├── pathway.py
   ├── simulator.py
   ├── datastore.py
   ├── allocator.py
   ├── app.py
   ├── dashboard/
   │   ├── app.py
   │   └── app_upgraded.py
   └── sample_data/
       └── kpetchem_demo.csv
   ```

3. **Install missing dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Dashboard versions

**Basic Dashboard**: `dashboard/app.py`
- Simpler interface
- Uses existing allocator.py
- Good for basic testing

**Advanced Dashboard**: `dashboard/app_upgraded.py` 
- Full 768-case Monte Carlo system
- Composite visualization with percentile ribbons
- Requires all upgraded modules

### 4. Module-specific errors

#### If `VectorizedBudgetAllocator` not found:
- You're using the old allocator
- Switch to basic dashboard or update imports

#### If `HighPerformanceSimulator` not found:
- Use the basic dashboard for now
- Or ensure simulator.py is the upgraded version

#### If `SimulationDataStore` not found:
- Use the basic dashboard
- Or ensure datastore.py exists

### 5. Python path issues

Add the kpetchem_budget directory to your Python path:

**On Windows**:
```cmd
set PYTHONPATH=%PYTHONPATH%;C:\path\to\kpetchem_budget
```

**On Mac/Linux**:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/kpetchem_budget"
```

### 6. Quick diagnostic

Run this diagnostic script:

```python
# diagnostic.py
import os
import sys
from pathlib import Path

print("=== K-PetChem Diagnostic ===")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

required_files = [
    'data_layer.py',
    'parameter_space.py', 
    'pathway.py',
    'simulator.py',
    'datastore.py'
]

print("\nChecking required files:")
for file in required_files:
    exists = Path(file).exists()
    print(f"  {file}: {'✅' if exists else '❌'}")

print("\nTrying imports:")
try:
    from data_layer import load_global_budget
    print("  data_layer: ✅")
except ImportError as e:
    print(f"  data_layer: ❌ {e}")

try:
    from parameter_space import ParameterGrid
    print("  parameter_space: ✅")
except ImportError as e:
    print(f"  parameter_space: ❌ {e}")

try:
    from simulator import HighPerformanceSimulator
    print("  simulator (upgraded): ✅")
except ImportError as e:
    print(f"  simulator (upgraded): ❌ {e}")
```

### 7. Running different dashboard versions

**For basic functionality**:
```bash
streamlit run app.py
```

**For advanced Monte Carlo system**:
```bash
streamlit run dashboard/app_upgraded.py
```

**Using the launcher**:
```bash
python run_dashboard.py
```

### 8. Common fixes

1. **Always run from the kpetchem_budget directory**
2. **Use absolute imports in production**
3. **Check Python version compatibility (3.11+ recommended)**
4. **Ensure all dependencies are installed**

### 9. If all else fails

1. **Clone fresh copy**
2. **Install requirements**: `pip install -r requirements.txt`
3. **Run diagnostic script**
4. **Use launcher script**: `python run_dashboard.py`

### 10. Environment setup

**Create virtual environment**:
```bash
python -m venv kpetchem_env
source kpetchem_env/bin/activate  # On Windows: kpetchem_env\Scripts\activate
pip install -r requirements.txt
```

**Verify installation**:
```bash
python -c "import streamlit, pandas, numpy, plotly; print('All dependencies OK')"
```

---

## Quick Start Commands

```bash
# Navigate to directory
cd kpetchem_budget/

# Install dependencies
pip install -r requirements.txt

# Run diagnostic
python -c "from data_layer import load_global_budget; print('Imports OK')"

# Launch dashboard
python run_dashboard.py
```

---

If you continue to have issues, please check:
1. File permissions
2. Python version (3.11+ recommended)
3. Virtual environment activation
4. All files are present and not corrupted