"""
Test cases for cohort vs parent/child model relationships.

Tests:
1. Grid search creates cohort with cohort_id
2. Cohort siblings have same cohort_id, different hyperparameters
3. Parent/child uses parent_model_id for feature evolution
4. Cohort and parent relationships can coexist
5. Database queries return correct cohort_size
6. Fingerprints are unique per cohort sibling
"""

import pytest
import json
import uuid
from datetime import datetime
from training_service.trainer import compute_fingerprint


class MockDB:
    """Mock database for testing."""
    
    def __init__(self):
        self.models = {}
        self.created_models = []
    
    def create_model(self, **kwargs):
        """Store model creation data."""
        model_id = kwargs.get('id', str(uuid.uuid4()))
        kwargs['id'] = model_id
        self.models[model_id] = kwargs
        self.created_models.append(kwargs)
        return model_id
    
    def get_model(self, model_id):
        """Get model by ID."""
        return self.models.get(model_id)
    
    def list_models(self):
        """List all models with cohort_size."""
        result = []
        for model in self.models.values():
            model_copy = model.copy()
            # Calculate cohort_size
            if model_copy.get('cohort_id'):
                cohort_size = sum(
                    1 for m in self.models.values() 
                    if m.get('cohort_id') == model_copy['cohort_id'] and m['id'] != model_copy['id']
                )
                model_copy['cohort_size'] = cohort_size
            else:
                model_copy['cohort_size'] = 0
            result.append(model_copy)
        return result
    
    def get_cohort_models(self, cohort_id):
        """Get all models in a cohort."""
        return [m for m in self.models.values() if m.get('cohort_id') == cohort_id]
    
    def get_children(self, parent_id):
        """Get all child models (feature evolution)."""
        return [m for m in self.models.values() if m.get('parent_model_id') == parent_id]


class TestCohortRelationships:
    """Test cohort relationship functionality."""
    
    def test_grid_search_creates_cohort(self):
        """Test that grid search models share a cohort_id."""
        db = MockDB()
        cohort_id = str(uuid.uuid4())
        
        # Simulate grid search creating 4 models
        hyperparams_grid = [
            {'alpha': 0.1, 'l1_ratio': 0.5},
            {'alpha': 0.1, 'l1_ratio': 0.7},
            {'alpha': 1.0, 'l1_ratio': 0.5},
            {'alpha': 1.0, 'l1_ratio': 0.7},
        ]
        
        model_ids = []
        for hp in hyperparams_grid:
            model_id = db.create_model(
                cohort_id=cohort_id,
                is_grid_member=True,
                hyperparameters=json.dumps(hp),
                parent_model_id=None,  # No parent for pure grid search
                symbol='AAPL',
                algorithm='Ridge'
            )
            model_ids.append(model_id)
        
        # Verify all models share cohort_id
        cohort_models = db.get_cohort_models(cohort_id)
        assert len(cohort_models) == 4, "Should have 4 cohort siblings"
        
        for model in cohort_models:
            assert model['cohort_id'] == cohort_id
            assert model['is_grid_member'] is True
            assert model['parent_model_id'] is None
            assert model['hyperparameters'] is not None
        
        print("✅ Grid search creates cohort with shared cohort_id")
    
    def test_cohort_size_calculation(self):
        """Test that cohort_size is calculated correctly."""
        db = MockDB()
        cohort_id = str(uuid.uuid4())
        
        # Create 3 models in cohort
        for i in range(3):
            db.create_model(
                cohort_id=cohort_id,
                is_grid_member=True,
                hyperparameters=json.dumps({'alpha': 0.1 * (i+1)}),
                symbol='AAPL'
            )
        
        # List models and check cohort_size
        models = db.list_models()
        for model in models:
            if model['cohort_id'] == cohort_id:
                # Each model should see 2 siblings (3 total - self)
                assert model['cohort_size'] == 2, f"Expected cohort_size=2, got {model['cohort_size']}"
        
        print("✅ Cohort size calculated correctly (excludes self)")
    
    def test_parent_child_feature_evolution(self):
        """Test that parent/child relationship works for feature evolution."""
        db = MockDB()
        
        # Create parent model with 100 features
        parent_id = db.create_model(
            symbol='AAPL',
            algorithm='Ridge',
            columns_initial=100,
            columns_remaining=100,
            parent_model_id=None,
            cohort_id=None,  # Not part of grid search
            is_grid_member=False
        )
        
        # Create child model with pruned features
        child_id = db.create_model(
            symbol='AAPL',
            algorithm='Ridge',
            columns_initial=100,
            columns_remaining=50,  # Pruned to 50 features
            parent_model_id=parent_id,
            cohort_id=None,  # Not part of grid search
            is_grid_member=False
        )
        
        # Verify relationship
        children = db.get_children(parent_id)
        assert len(children) == 1, "Should have 1 child"
        assert children[0]['id'] == child_id
        assert children[0]['parent_model_id'] == parent_id
        assert children[0]['columns_remaining'] == 50
        assert children[0]['cohort_id'] is None
        
        print("✅ Parent/child relationship works for feature evolution")
    
    def test_cohort_and_parent_coexist(self):
        """Test that a model can have both cohort_id and parent_model_id."""
        db = MockDB()
        
        # Scenario: Grid search on pruned features
        # 1. Create parent model (feature evolution source)
        parent_id = db.create_model(
            symbol='AAPL',
            algorithm='Ridge',
            columns_initial=100,
            columns_remaining=50,
            is_grid_member=False
        )
        
        # 2. Create cohort from grid search on pruned model
        cohort_id = str(uuid.uuid4())
        hyperparams = [
            {'alpha': 0.1, 'l1_ratio': 0.5},
            {'alpha': 1.0, 'l1_ratio': 0.5},
        ]
        
        for hp in hyperparams:
            db.create_model(
                cohort_id=cohort_id,           # Part of cohort
                parent_model_id=parent_id,     # Evolved from parent
                is_grid_member=True,
                hyperparameters=json.dumps(hp),
                columns_initial=100,
                columns_remaining=50,          # Same as parent
                symbol='AAPL',
                algorithm='Ridge'
            )
        
        # Verify both relationships exist
        cohort_models = db.get_cohort_models(cohort_id)
        assert len(cohort_models) == 2, "Should have 2 cohort siblings"
        
        for model in cohort_models:
            assert model['cohort_id'] == cohort_id, "Should be in cohort"
            assert model['parent_model_id'] == parent_id, "Should have parent"
            assert model['is_grid_member'] is True
        
        # Also check from parent perspective
        children = db.get_children(parent_id)
        assert len(children) == 2, "Parent should have 2 children"
        
        print("✅ Cohort and parent relationships coexist correctly")
    
    def test_unique_fingerprints_per_cohort_sibling(self):
        """Test that each cohort sibling has unique fingerprint."""
        cohort_id = str(uuid.uuid4())
        
        # Base config (same for all siblings)
        base_config = {
            'symbol': 'AAPL',
            'algorithm': 'Ridge',
            'target_col': 'target_5',
            'timeframe': '1h',
            'target_transform': 'log',
            'data_options': {'lookback': 60},
            'parent_model_id': None,
            'grid_search_cv': 5,
            'grid_search_scoring': 'r2'
        }
        
        # Different hyperparameters per sibling
        hyperparams_list = [
            {'alpha': 0.1, 'l1_ratio': 0.5},
            {'alpha': 0.1, 'l1_ratio': 0.7},
            {'alpha': 1.0, 'l1_ratio': 0.5},
        ]
        
        fingerprints = []
        for hp in hyperparams_list:
            fp = compute_fingerprint(
                **base_config,
                hyperparameters=hp
            )
            fingerprints.append(fp)
        
        # All fingerprints should be unique
        assert len(fingerprints) == len(set(fingerprints)), "Fingerprints should be unique"
        
        # Verify same config produces same fingerprint
        fp_duplicate = compute_fingerprint(**base_config, hyperparameters=hyperparams_list[0])
        assert fp_duplicate == fingerprints[0], "Same config should produce same fingerprint"
        
        print("✅ Each cohort sibling has unique fingerprint")
    
    def test_cohort_without_parent(self):
        """Test pure grid search cohort (no parent_model_id)."""
        db = MockDB()
        cohort_id = str(uuid.uuid4())
        
        # Create 2 models in cohort without parent
        for i in range(2):
            db.create_model(
                cohort_id=cohort_id,
                parent_model_id=None,  # No parent
                is_grid_member=True,
                hyperparameters=json.dumps({'alpha': 0.1 * (i+1)}),
                symbol='AAPL'
            )
        
        cohort_models = db.get_cohort_models(cohort_id)
        assert len(cohort_models) == 2
        
        for model in cohort_models:
            assert model['parent_model_id'] is None, "Pure cohort should have no parent"
            assert model['cohort_id'] == cohort_id
        
        print("✅ Pure cohort works without parent_model_id")
    
    def test_parent_without_cohort(self):
        """Test parent/child relationship without cohort."""
        db = MockDB()
        
        # Parent model
        parent_id = db.create_model(
            symbol='AAPL',
            algorithm='Ridge',
            parent_model_id=None,
            cohort_id=None,
            is_grid_member=False,
            columns_remaining=100
        )
        
        # Child model (feature evolution)
        child_id = db.create_model(
            symbol='AAPL',
            algorithm='Ridge',
            parent_model_id=parent_id,
            cohort_id=None,  # No cohort
            is_grid_member=False,
            columns_remaining=50
        )
        
        children = db.get_children(parent_id)
        assert len(children) == 1
        assert children[0]['cohort_id'] is None, "Feature evolution child should have no cohort"
        assert children[0]['parent_model_id'] == parent_id
        
        print("✅ Parent/child works without cohort_id")
    
    def test_multiple_cohorts_distinct(self):
        """Test that multiple cohorts remain distinct."""
        db = MockDB()
        
        # Create two separate cohorts
        cohort1_id = str(uuid.uuid4())
        cohort2_id = str(uuid.uuid4())
        
        # Cohort 1: AAPL Ridge
        for i in range(2):
            db.create_model(
                cohort_id=cohort1_id,
                symbol='AAPL',
                algorithm='Ridge',
                hyperparameters=json.dumps({'alpha': 0.1 * (i+1)}),
                is_grid_member=True
            )
        
        # Cohort 2: GOOGL Lasso
        for i in range(3):
            db.create_model(
                cohort_id=cohort2_id,
                symbol='GOOGL',
                algorithm='Lasso',
                hyperparameters=json.dumps({'alpha': 1.0 * (i+1)}),
                is_grid_member=True
            )
        
        # Verify cohorts are distinct
        cohort1_models = db.get_cohort_models(cohort1_id)
        cohort2_models = db.get_cohort_models(cohort2_id)
        
        assert len(cohort1_models) == 2, "Cohort 1 should have 2 models"
        assert len(cohort2_models) == 3, "Cohort 2 should have 3 models"
        
        # Check no overlap
        cohort1_ids = {m['id'] for m in cohort1_models}
        cohort2_ids = {m['id'] for m in cohort2_models}
        assert cohort1_ids.isdisjoint(cohort2_ids), "Cohorts should not overlap"
        
        print("✅ Multiple cohorts remain distinct")
    
    def test_fingerprint_includes_cohort_id(self):
        """Test that fingerprint changes when cohort_id differs (if parent_model_id is set)."""
        # Note: cohort_id itself isn't in fingerprint, but parent_model_id is
        # This tests the distinction between cohort and parent relationships
        
        base_config = {
            'symbol': 'AAPL',
            'algorithm': 'Ridge',
            'target_col': 'target_5',
            'timeframe': '1h',
            'target_transform': 'log',
            'data_options': {'lookback': 60},
            'hyperparameters': {'alpha': 0.1},
            'grid_search_cv': 5,
            'grid_search_scoring': 'r2'
        }
        
        # Fingerprint without parent
        fp_no_parent = compute_fingerprint(**base_config, parent_model_id=None)
        
        # Fingerprint with parent
        parent_id = str(uuid.uuid4())
        fp_with_parent = compute_fingerprint(**base_config, parent_model_id=parent_id)
        
        # Should be different
        assert fp_no_parent != fp_with_parent, "Fingerprint should differ with parent_model_id"
        
        print("✅ Fingerprint includes parent_model_id in hash")


class TestCohortQueries:
    """Test database query patterns for cohorts."""
    
    def test_find_all_siblings(self):
        """Test querying all siblings in a cohort."""
        db = MockDB()
        cohort_id = str(uuid.uuid4())
        
        # Create cohort with varying hyperparameters
        expected_alphas = [0.01, 0.1, 1.0, 10.0]
        for alpha in expected_alphas:
            db.create_model(
                cohort_id=cohort_id,
                hyperparameters=json.dumps({'alpha': alpha, 'l1_ratio': 0.5}),
                is_grid_member=True,
                symbol='AAPL'
            )
        
        # Query siblings
        siblings = db.get_cohort_models(cohort_id)
        assert len(siblings) == 4
        
        # Extract and verify alphas
        found_alphas = [
            json.loads(m['hyperparameters'])['alpha'] 
            for m in siblings
        ]
        assert sorted(found_alphas) == sorted(expected_alphas)
        
        print("✅ Can query all siblings in cohort")
    
    def test_cohort_leader_identification(self):
        """Test identifying the 'leader' or representative of a cohort."""
        db = MockDB()
        cohort_id = str(uuid.uuid4())
        
        # Create models with different metrics
        best_r2 = None
        for i, r2 in enumerate([0.75, 0.92, 0.88, 0.80]):  # 2nd is best
            model_id = db.create_model(
                cohort_id=cohort_id,
                hyperparameters=json.dumps({'alpha': 0.1 * (i+1)}),
                metrics=json.dumps({'r2': r2}),
                is_grid_member=True,
                symbol='AAPL'
            )
            if r2 == 0.92:
                best_r2 = model_id
        
        # Find best model in cohort
        siblings = db.get_cohort_models(cohort_id)
        best_model = max(
            siblings, 
            key=lambda m: json.loads(m.get('metrics', '{}') or '{}').get('r2', 0)
        )
        
        assert best_model['id'] == best_r2
        assert json.loads(best_model['metrics'])['r2'] == 0.92
        
        print("✅ Can identify best model in cohort")
    
    def test_list_all_cohorts(self):
        """Test listing all unique cohorts."""
        db = MockDB()
        
        # Create 3 different cohorts
        cohort_ids = [str(uuid.uuid4()) for _ in range(3)]
        cohort_sizes = [2, 3, 4]
        
        for cohort_id, size in zip(cohort_ids, cohort_sizes):
            for i in range(size):
                db.create_model(
                    cohort_id=cohort_id,
                    is_grid_member=True,
                    hyperparameters=json.dumps({'alpha': 0.1 * (i+1)}),
                    symbol='AAPL'
                )
        
        # Get unique cohorts
        all_models = db.list_models()
        unique_cohorts = {m['cohort_id'] for m in all_models if m.get('cohort_id')}
        
        assert len(unique_cohorts) == 3, "Should have 3 distinct cohorts"
        assert unique_cohorts == set(cohort_ids)
        
        print("✅ Can list all unique cohorts")


def run_all_tests():
    """Run all test suites."""
    print("\n" + "="*60)
    print("COHORT VS PARENT/CHILD RELATIONSHIP TESTS")
    print("="*60 + "\n")
    
    # Test cohort relationships
    print("--- Testing Cohort Relationships ---")
    suite1 = TestCohortRelationships()
    suite1.test_grid_search_creates_cohort()
    suite1.test_cohort_size_calculation()
    suite1.test_parent_child_feature_evolution()
    suite1.test_cohort_and_parent_coexist()
    suite1.test_unique_fingerprints_per_cohort_sibling()
    suite1.test_cohort_without_parent()
    suite1.test_parent_without_cohort()
    suite1.test_multiple_cohorts_distinct()
    suite1.test_fingerprint_includes_cohort_id()
    
    print("\n--- Testing Cohort Queries ---")
    suite2 = TestCohortQueries()
    suite2.test_find_all_siblings()
    suite2.test_cohort_leader_identification()
    suite2.test_list_all_cohorts()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
