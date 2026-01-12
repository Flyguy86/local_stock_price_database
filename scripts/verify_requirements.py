#!/usr/bin/env python3
"""
Verify all Python imports across the codebase are covered in requirements.txt
"""
import ast
import sys
from pathlib import Path
from collections import defaultdict

# Known standard library modules (partial list)
STDLIB = {
    'os', 'sys', 'pathlib', 'datetime', 'time', 'json', 'logging', 'uuid',
    'threading', 'multiprocessing', 'asyncio', 'contextlib', 'functools',
    'itertools', 'collections', 'typing', 'traceback', 're', 'io', 'tempfile',
    'shutil', 'subprocess', 'argparse', 'configparser', 'enum'
}

def extract_imports(filepath):
    """Extract all import statements from a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Get root package (e.g., 'sklearn' from 'sklearn.ensemble')
                    root = alias.name.split('.')[0]
                    imports.add(root)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split('.')[0]
                    imports.add(root)
        
        return imports
    except Exception as e:
        print(f"Error parsing {filepath}: {e}", file=sys.stderr)
        return set()

def scan_project(project_root):
    """Scan all Python files and collect imports."""
    all_imports = defaultdict(list)
    
    for py_file in Path(project_root).rglob("*.py"):
        # Skip venv, __pycache__, etc.
        if any(skip in str(py_file) for skip in ['venv', '__pycache__', '.git', 'build', 'dist']):
            continue
        
        imports = extract_imports(py_file)
        for imp in imports:
            all_imports[imp].append(str(py_file))
    
    return all_imports

def load_requirements(req_file):
    """Parse requirements.txt and extract package names."""
    packages = set()
    try:
        with open(req_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Extract package name (before >=, ==, <, etc.)
                pkg = line.split('>=')[0].split('==')[0].split('<')[0].split('[')[0].strip()
                # Normalize package names (e.g., pandas-ta -> pandas_ta for import check)
                packages.add(pkg.replace('-', '_'))
                packages.add(pkg)  # Also keep original
    except FileNotFoundError:
        print(f"Requirements file not found: {req_file}", file=sys.stderr)
    
    return packages

# Map common import names to PyPI package names
IMPORT_TO_PACKAGE = {
    'sklearn': 'scikit-learn',
    'cv2': 'opencv-python',
    'PIL': 'Pillow',
    'yaml': 'PyYAML',
    'dotenv': 'python-dotenv',
    'dateutil': 'python-dateutil',
    'talib': 'ta-lib',
    'pandas_ta': 'pandas-ta',
    'alpaca_trade_api': 'alpaca-trade-api',
}

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    req_file = project_root / "requirements.txt"
    
    print("🔍 Scanning project for imports...")
    all_imports = scan_project(project_root)
    
    print(f"📦 Loading requirements from {req_file}...")
    required_packages = load_requirements(req_file)
    
    # Filter to third-party imports only
    third_party = {
        imp for imp in all_imports.keys()
        if imp not in STDLIB and not imp.startswith('app') and not imp.startswith('feature_service')
        and not imp.startswith('training_service') and not imp.startswith('simulation_service')
        and not imp.startswith('optimization_service')
    }
    
    print(f"\n✅ Found {len(third_party)} unique third-party imports\n")
    
    # Check coverage
    missing = []
    for imp in sorted(third_party):
        # Map import name to package name
        pkg_name = IMPORT_TO_PACKAGE.get(imp, imp)
        
        # Check if package or its normalized form is in requirements
        if pkg_name not in required_packages and imp not in required_packages:
            missing.append(imp)
            files = all_imports[imp][:3]  # Show first 3 files
            print(f"⚠️  Missing: {imp} (package: {pkg_name})")
            for f in files:
                print(f"     Used in: {f}")
    
    if not missing:
        print("✅ All imports are covered in requirements.txt")
    else:
        print(f"\n❌ {len(missing)} packages missing from requirements.txt")
        sys.exit(1)
