#!/usr/bin/env python3
"""Analyze storage usage across DuckDB and Parquet."""

import os
from pathlib import Path


def get_size(path):
    """Get file or directory size in bytes."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total += os.path.getsize(filepath)
    return total


def format_size(bytes_size):
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def main():
    data_dir = Path("/workspaces/local_stock_price_database/data")
    
    print("=" * 80)
    print("STORAGE ANALYSIS")
    print("=" * 80)
    
    # DuckDB breakdown
    print("\n📊 DuckDB Files (/data/duckdb):")
    print("-" * 80)
    duckdb_dir = data_dir / "duckdb"
    duckdb_files = []
    if duckdb_dir.exists():
        for file in sorted(duckdb_dir.iterdir()):
            if file.is_file():
                size = file.stat().st_size
                duckdb_files.append((file.name, size))
                print(f"  {file.name:25s} {format_size(size):>12s}")
    
    total_duckdb = sum(size for _, size in duckdb_files)
    print(f"  {'TOTAL':25s} {format_size(total_duckdb):>12s}")
    
    # Parquet breakdown
    print("\n📦 Parquet Files (/data/parquet):")
    print("-" * 80)
    parquet_dir = data_dir / "parquet"
    if parquet_dir.exists():
        parquet_symbols = []
        for symbol_dir in sorted(parquet_dir.iterdir()):
            if symbol_dir.is_dir():
                size = get_size(symbol_dir)
                parquet_symbols.append((symbol_dir.name, size))
                print(f"  {symbol_dir.name:25s} {format_size(size):>12s}")
    
    total_parquet = sum(size for _, size in parquet_symbols)
    print(f"  {'TOTAL':25s} {format_size(total_parquet):>12s}")
    
    # Comparison
    print("\n📈 Size Comparison:")
    print("-" * 80)
    print(f"  DuckDB:  {format_size(total_duckdb):>12s}")
    print(f"  Parquet: {format_size(total_parquet):>12s}")
    if total_parquet > 0:
        ratio = total_duckdb / total_parquet
        print(f"  Ratio:   {ratio:>12.1f}x (DuckDB / Parquet)")
    
    # Explain the difference
    print("\n💡 Why is DuckDB larger?")
    print("-" * 80)
    print("  1. Multiple database files:")
    print("     - local.db (main ingestion data)")
    print("     - features.db (engineered features)")
    print("     - models.db, optimization.db (training metadata)")
    print("     - Backup files (.backup, .v0.9)")
    print("     - Write-ahead log (.wal)")
    print()
    print("  2. Storage overhead:")
    print("     - DuckDB includes indexes for fast queries")
    print("     - Transaction logs for ACID compliance")
    print("     - Internal metadata and statistics")
    print()
    print("  3. Compression:")
    print("     - Parquet uses columnar compression (very efficient)")
    print("     - DuckDB row storage is less compressed")
    print()
    print("  4. Redundancy:")
    print("     - local.db contains raw OHLCV bars")
    print("     - features.db may duplicate data + engineered features")
    print("     - Parquet only contains raw bars (source of truth)")
    
    # Recommendations
    print("\n🔧 Optimization Recommendations:")
    print("-" * 80)
    
    # Check for large backup files
    backup_size = 0
    for file, size in duckdb_files:
        if 'backup' in file or 'v0.9' in file:
            backup_size += size
    
    if backup_size > 0:
        print(f"  ⚠️  Remove old backups: {format_size(backup_size)} can be freed")
        print("     rm /workspaces/local_stock_price_database/data/duckdb/*.backup")
        print("     rm /workspaces/local_stock_price_database/data/duckdb/*.v0.9")
    
    # Check WAL size
    wal_size = next((size for file, size in duckdb_files if file.endswith('.wal')), 0)
    if wal_size > 100 * 1024 * 1024:  # > 100MB
        print(f"  ⚠️  Large WAL file: {format_size(wal_size)}")
        print("     Run CHECKPOINT to flush WAL to main database")
    
    # Check if features.db exists
    features_size = next((size for file, size in duckdb_files if file == 'features.db'), 0)
    if features_size > total_parquet:
        print(f"  ⚠️  features.db ({format_size(features_size)}) is larger than raw Parquet")
        print("     This is expected if it contains many engineered features")
        print("     Consider using walk-forward folds instead (on-demand feature calculation)")
    
    print("\n✅ Current architecture (walk-forward folds):")
    print("   - Raw data: Parquet (224M, efficient columnar storage)")
    print("   - Queries: DuckDB (2.2G, fast indexed lookups)")
    print("   - Features: Generated on-demand per fold (prevents look-ahead bias)")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
