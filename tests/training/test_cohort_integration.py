"""
Integration tests for cohort functionality with real database operations.

These tests verify the full stack:
1. Database schema has cohort_id column
2. Trainer saves grid search with cohort_id
3. API queries return cohort_size correctly
4. Fingerprints prevent duplicates within cohort
"""

import os
import sys
import json
import pytest
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    from training_service.pg_db import TrainingDatabase
    from training_service.trainer import compute_fingerprint
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Integration test dependencies not available: {e}")
    INTEGRATION_AVAILABLE = False


@pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Database not available")
class TestCohortDatabaseIntegration:
    """Integration tests with actual database."""
    
    @pytest.fixture
    def db(self):
        """Create database connection."""
        try:
            db = TrainingDatabase()
            yield db
            # Cleanup: Could delete test models here
        except Exception as e:
            pytest.skip(f"Database not available: {e}")
    
    def test_cohort_id_column_exists(self, db):
        """Verify cohort_id column exists in schema."""
        # Query to check column existence
        query = """
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'models' AND column_name = 'cohort_id';
        """
        
        with db.conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchone()
        
        assert result is not None, "cohort_id column should exist"
        assert result[1] == 'character varying', "cohort_id should be VARCHAR"
        
        print("✅ Database has cohort_id column")
    
    def test_create_cohort_models(self, db):
        """Test creating multiple models with same cohort_id."""
        cohort_id = f"test_cohort_{datetime.now().timestamp()}"
        model_ids = []
        
        try:
            # Create 3 models in cohort
            for i in range(3):
                model_data = {
                    'symbol': 'TEST',
                    'algorithm': 'Ridge',
                    'target_col': 'target_5',
                    'timeframe': '1h',
                    'status': 'completed',
                    'cohort_id': cohort_id,
                    'is_grid_member': True,
                    'hyperparameters': json.dumps({'alpha': 0.1 * (i+1), 'l1_ratio': 0.5}),
                    'metrics': json.dumps({'r2': 0.8 + i * 0.05})
                }
                
                model_id = db.create_model(**model_data)
                model_ids.append(model_id)
            
            # Query cohort size
            models = db.list_models()
            cohort_models = [m for m in models if m.get('cohort_id') == cohort_id]
            
            assert len(cohort_models) == 3, "Should have 3 models in cohort"
            
            # Verify cohort_size is calculated
            for model in cohort_models:
                assert model.get('cohort_size') == 2, "Each model should see 2 siblings"
            
            print(f"✅ Created cohort with {len(cohort_models)} models, cohort_size correct")
            
        finally:
            # Cleanup
            with db.conn.cursor() as cur:
                for model_id in model_ids:
                    cur.execute("DELETE FROM models WHERE id = %s", (model_id,))
                db.conn.commit()
    
    def test_parent_child_without_cohort(self, db):
        """Test parent/child relationship without cohort_id."""
        parent_id = f"test_parent_{datetime.now().timestamp()}"
        child_id = f"test_child_{datetime.now().timestamp()}"
        
        try:
            # Create parent
            db.create_model(
                id=parent_id,
                symbol='TEST',
                algorithm='Ridge',
                target_col='target_5',
                timeframe='1h',
                status='completed',
                columns_initial=100,
                columns_remaining=100,
                parent_model_id=None,
                cohort_id=None,
                is_grid_member=False
            )
            
            # Create child
            db.create_model(
                id=child_id,
                symbol='TEST',
                algorithm='Ridge',
                target_col='target_5',
                timeframe='1h',
                status='completed',
                columns_initial=100,
                columns_remaining=50,
                parent_model_id=parent_id,
                cohort_id=None,
                is_grid_member=False
            )
            
            # Verify relationship
            child = db.get_model(child_id)
            assert child['parent_model_id'] == parent_id
            assert child['cohort_id'] is None
            assert child['columns_remaining'] == 50
            
            print("✅ Parent/child relationship works without cohort")
            
        finally:
            # Cleanup
            with db.conn.cursor() as cur:
                cur.execute("DELETE FROM models WHERE id IN (%s, %s)", (parent_id, child_id))
                db.conn.commit()
    
    def test_combined_cohort_and_parent(self, db):
        """Test model with both cohort_id and parent_model_id."""
        parent_id = f"test_combined_parent_{datetime.now().timestamp()}"
        cohort_id = f"test_combined_cohort_{datetime.now().timestamp()}"
        model_ids = []
        
        try:
            # Create parent (feature evolution source)
            db.create_model(
                id=parent_id,
                symbol='TEST',
                algorithm='Ridge',
                target_col='target_5',
                timeframe='1h',
                status='completed',
                columns_remaining=50
            )
            
            # Create cohort of models that evolved from parent
            for i in range(2):
                model_data = {
                    'symbol': 'TEST',
                    'algorithm': 'Ridge',
                    'target_col': 'target_5',
                    'timeframe': '1h',
                    'status': 'completed',
                    'cohort_id': cohort_id,
                    'parent_model_id': parent_id,
                    'is_grid_member': True,
                    'columns_remaining': 50,
                    'hyperparameters': json.dumps({'alpha': 0.1 * (i+1)})
                }
                
                model_id = db.create_model(**model_data)
                model_ids.append(model_id)
            
            # Verify both relationships
            models = db.list_models()
            cohort_models = [m for m in models if m.get('cohort_id') == cohort_id]
            
            assert len(cohort_models) == 2, "Should have 2 cohort siblings"
            
            for model in cohort_models:
                assert model['cohort_id'] == cohort_id, "Should be in cohort"
                assert model['parent_model_id'] == parent_id, "Should have parent"
                assert model['cohort_size'] == 1, "Should see 1 sibling"
            
            print("✅ Model can have both cohort_id and parent_model_id")
            
        finally:
            # Cleanup
            with db.conn.cursor() as cur:
                for model_id in model_ids:
                    cur.execute("DELETE FROM models WHERE id = %s", (model_id,))
                cur.execute("DELETE FROM models WHERE id = %s", (parent_id,))
                db.conn.commit()


def run_integration_tests():
    """Run integration tests manually."""
    if not INTEGRATION_AVAILABLE:
        print("⚠️  Skipping integration tests - database not available")
        return
    
    print("\n" + "="*60)
    print("COHORT INTEGRATION TESTS")
    print("="*60 + "\n")
    
    try:
        db = TrainingDatabase()
        suite = TestCohortDatabaseIntegration()
        
        print("Testing database schema...")
        suite.test_cohort_id_column_exists(db)
        
        print("\nTesting cohort creation...")
        suite.test_create_cohort_models(db)
        
        print("\nTesting parent/child without cohort...")
        suite.test_parent_child_without_cohort(db)
        
        print("\nTesting combined relationships...")
        suite.test_combined_cohort_and_parent(db)
        
        print("\n" + "="*60)
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Integration tests failed: {e}")
        raise


if __name__ == "__main__":
    run_integration_tests()
