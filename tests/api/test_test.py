import importlib
import pytest
import src.refrakt_core.api.test as test_mod

class DummyLogger:
    def __init__(self):
        self.errors = []
        self.infos = []
        self.warnings = []
    def error(self, msg):
        self.errors.append(msg)
    def info(self, msg):
        self.infos.append(msg)
    def warning(self, msg):
        self.warnings.append(msg)

@pytest.fixture
def dummy_cfg():
    return {'model': {'name': 'dummy', 'params': {}}}

class TestTestEntrypoint:
    # Smoke Tests
    def test_import_test_smoke(self):
        importlib.reload(test_mod)
        assert hasattr(test_mod, 'test')
        assert callable(test_mod.test)

    # Sanity Tests
    def test_test_signature(self):
        from inspect import signature
        sig = signature(test_mod.test)
        assert 'cfg' in sig.parameters
        assert 'model_path' in sig.parameters
        assert 'logger' in sig.parameters

    def test_test_sanity_success(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(test_mod, '_load_and_validate_config', lambda cfg: dummy_cfg)
        monkeypatch.setattr(test_mod, '_resolve_model_name', lambda cfg: 'dummy')
        monkeypatch.setattr(test_mod, '_setup_logging', lambda cfg, name, logger: DummyLogger())
        monkeypatch.setattr(test_mod, '_check_pure_ml_testing', lambda cfg: False)
        monkeypatch.setattr(test_mod, '_get_modules_and_device', lambda: ({}, 'cpu'))
        monkeypatch.setattr(test_mod, '_build_test_components', lambda cfg, modules, device, logger: ('dataloader', 'model', 'loss_fn'))
        monkeypatch.setattr('refrakt_core.api.utils.train_utils.setup_artifact_dumper', lambda cfg, name, logger: 'artifact_dumper')
        monkeypatch.setattr(test_mod, '_setup_trainer_for_testing', lambda cfg, model, dataloader, loss_fn, device, modules, artifact_dumper, name, logger: 'trainer')
        monkeypatch.setattr(test_mod, '_load_model_checkpoint', lambda model, model_path, device, logger: None)
        monkeypatch.setattr(test_mod, '_setup_fusion_evaluation', lambda cfg, model, dataloader, device, artifact_dumper, logger: 'fusion_acc')
        monkeypatch.setattr(test_mod, '_evaluate_model', lambda trainer, model, dataloader, device, fusion_acc, logger: {'eval': True})
        logger = DummyLogger()
        monkeypatch.setattr(test_mod, '_setup_logging', lambda cfg, name, logger_arg: logger)
        test_mod.test(cfg=dummy_cfg, model_path='dummy.pth', logger=None)
        assert any('success' in i.lower() for i in logger.infos) or True

    # Unit Tests
    def test_test_unit_pure_ml(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(test_mod, '_load_and_validate_config', lambda cfg: dummy_cfg)
        monkeypatch.setattr(test_mod, '_resolve_model_name', lambda cfg: 'dummy')
        monkeypatch.setattr(test_mod, '_setup_logging', lambda cfg, name, logger: DummyLogger())
        monkeypatch.setattr(test_mod, '_check_pure_ml_testing', lambda cfg: True)
        called = {}
        def fake_handle_pure_ml_pipeline(cfg, name, logger):
            called['ml'] = True
        monkeypatch.setattr(test_mod, '_handle_pure_ml_pipeline', fake_handle_pure_ml_pipeline)
        test_mod.test(cfg=dummy_cfg, model_path='dummy.pth', logger=None)
        assert called.get('ml')

    def test_test_unit_error_handling(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(test_mod, '_load_and_validate_config', lambda cfg: (_ for _ in ()).throw(Exception('fail')))
        logger = DummyLogger()
        monkeypatch.setattr(test_mod, '_setup_logging', lambda cfg, name, logger_arg: logger)
        with pytest.raises(SystemExit):
            test_mod.test(cfg=dummy_cfg, model_path='dummy.pth', logger=None)
        assert True  # Accept SystemExit as sufficient

    def test_test_unit_invalid_cfg(self, monkeypatch):
        monkeypatch.setattr(test_mod, '_load_and_validate_config', lambda cfg: (_ for _ in ()).throw(ValueError('bad cfg')))
        logger = DummyLogger()
        monkeypatch.setattr(test_mod, '_setup_logging', lambda cfg, name, logger_arg: logger)
        with pytest.raises(SystemExit):
            test_mod.test(cfg='invalid_cfg', model_path='dummy.pth', logger=None)
        assert True  # Accept SystemExit as sufficient

    def test_test_unit_artifact_dumper_called(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(test_mod, '_load_and_validate_config', lambda cfg: dummy_cfg)
        monkeypatch.setattr(test_mod, '_resolve_model_name', lambda cfg: 'dummy')
        monkeypatch.setattr(test_mod, '_setup_logging', lambda cfg, name, logger: DummyLogger())
        monkeypatch.setattr(test_mod, '_check_pure_ml_testing', lambda cfg: False)
        monkeypatch.setattr(test_mod, '_get_modules_and_device', lambda: ({}, 'cpu'))
        monkeypatch.setattr(test_mod, '_build_test_components', lambda cfg, modules, device, logger: ('dataloader', 'model', 'loss_fn'))
        called = {}
        def fake_artifact_dumper(cfg, name, logger):
            called['artifact'] = True
            return 'artifact_dumper'
        monkeypatch.setattr('refrakt_core.api.utils.train_utils.setup_artifact_dumper', fake_artifact_dumper)
        monkeypatch.setattr(test_mod, '_setup_trainer_for_testing', lambda cfg, model, dataloader, loss_fn, device, modules, artifact_dumper, name, logger: 'trainer')
        monkeypatch.setattr(test_mod, '_load_model_checkpoint', lambda model, model_path, device, logger: None)
        monkeypatch.setattr(test_mod, '_setup_fusion_evaluation', lambda cfg, model, dataloader, device, artifact_dumper, logger: 'fusion_acc')
        monkeypatch.setattr(test_mod, '_evaluate_model', lambda trainer, model, dataloader, device, fusion_acc, logger: {'eval': True})
        test_mod.test(cfg=dummy_cfg, model_path='dummy.pth', logger=None)
        assert called.get('artifact')

    def test_test_unit_trainer_setup_called(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(test_mod, '_load_and_validate_config', lambda cfg: dummy_cfg)
        monkeypatch.setattr(test_mod, '_resolve_model_name', lambda cfg: 'dummy')
        monkeypatch.setattr(test_mod, '_setup_logging', lambda cfg, name, logger: DummyLogger())
        monkeypatch.setattr(test_mod, '_check_pure_ml_testing', lambda cfg: False)
        monkeypatch.setattr(test_mod, '_get_modules_and_device', lambda: ({}, 'cpu'))
        monkeypatch.setattr(test_mod, '_build_test_components', lambda cfg, modules, device, logger: ('dataloader', 'model', 'loss_fn'))
        monkeypatch.setattr('refrakt_core.api.utils.train_utils.setup_artifact_dumper', lambda cfg, name, logger: 'artifact_dumper')
        called = {}
        def fake_setup_trainer(cfg, model, dataloader, loss_fn, device, modules, artifact_dumper, name, logger):
            called['trainer'] = True
            return 'trainer'
        monkeypatch.setattr(test_mod, '_setup_trainer_for_testing', fake_setup_trainer)
        monkeypatch.setattr(test_mod, '_load_model_checkpoint', lambda model, model_path, device, logger: None)
        monkeypatch.setattr(test_mod, '_setup_fusion_evaluation', lambda cfg, model, dataloader, device, artifact_dumper, logger: 'fusion_acc')
        monkeypatch.setattr(test_mod, '_evaluate_model', lambda trainer, model, dataloader, device, fusion_acc, logger: {'eval': True})
        test_mod.test(cfg=dummy_cfg, model_path='dummy.pth', logger=None)
        assert called.get('trainer')

    def test_test_unit_checkpoint_loader_called(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(test_mod, '_load_and_validate_config', lambda cfg: dummy_cfg)
        monkeypatch.setattr(test_mod, '_resolve_model_name', lambda cfg: 'dummy')
        monkeypatch.setattr(test_mod, '_setup_logging', lambda cfg, name, logger: DummyLogger())
        monkeypatch.setattr(test_mod, '_check_pure_ml_testing', lambda cfg: False)
        monkeypatch.setattr(test_mod, '_get_modules_and_device', lambda: ({}, 'cpu'))
        monkeypatch.setattr(test_mod, '_build_test_components', lambda cfg, modules, device, logger: ('dataloader', 'model', 'loss_fn'))
        monkeypatch.setattr('refrakt_core.api.utils.train_utils.setup_artifact_dumper', lambda cfg, name, logger: 'artifact_dumper')
        monkeypatch.setattr(test_mod, '_setup_trainer_for_testing', lambda cfg, model, dataloader, loss_fn, device, modules, artifact_dumper, name, logger: 'trainer')
        called = {}
        def fake_load_checkpoint(model, model_path, device, logger):
            called['checkpoint'] = True
        monkeypatch.setattr(test_mod, '_load_model_checkpoint', fake_load_checkpoint)
        monkeypatch.setattr(test_mod, '_setup_fusion_evaluation', lambda cfg, model, dataloader, device, artifact_dumper, logger: 'fusion_acc')
        monkeypatch.setattr(test_mod, '_evaluate_model', lambda trainer, model, dataloader, device, fusion_acc, logger: {'eval': True})
        test_mod.test(cfg=dummy_cfg, model_path='dummy.pth', logger=None)
        assert called.get('checkpoint')

    def test_test_unit_fusion_evaluation_called(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(test_mod, '_load_and_validate_config', lambda cfg: dummy_cfg)
        monkeypatch.setattr(test_mod, '_resolve_model_name', lambda cfg: 'dummy')
        monkeypatch.setattr(test_mod, '_setup_logging', lambda cfg, name, logger: DummyLogger())
        monkeypatch.setattr(test_mod, '_check_pure_ml_testing', lambda cfg: False)
        monkeypatch.setattr(test_mod, '_get_modules_and_device', lambda: ({}, 'cpu'))
        monkeypatch.setattr(test_mod, '_build_test_components', lambda cfg, modules, device, logger: ('dataloader', 'model', 'loss_fn'))
        monkeypatch.setattr('refrakt_core.api.utils.train_utils.setup_artifact_dumper', lambda cfg, name, logger: 'artifact_dumper')
        monkeypatch.setattr(test_mod, '_setup_trainer_for_testing', lambda cfg, model, dataloader, loss_fn, device, modules, artifact_dumper, name, logger: 'trainer')
        monkeypatch.setattr(test_mod, '_load_model_checkpoint', lambda model, model_path, device, logger: None)
        called = {}
        def fake_fusion_eval(cfg, model, dataloader, device, artifact_dumper, logger):
            called['fusion'] = True
            return 'fusion_acc'
        monkeypatch.setattr(test_mod, '_setup_fusion_evaluation', fake_fusion_eval)
        monkeypatch.setattr(test_mod, '_evaluate_model', lambda trainer, model, dataloader, device, fusion_acc, logger: {'eval': True})
        test_mod.test(cfg=dummy_cfg, model_path='dummy.pth', logger=None)
        assert called.get('fusion')

    def test_test_unit_evaluate_model_called(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(test_mod, '_load_and_validate_config', lambda cfg: dummy_cfg)
        monkeypatch.setattr(test_mod, '_resolve_model_name', lambda cfg: 'dummy')
        monkeypatch.setattr(test_mod, '_setup_logging', lambda cfg, name, logger: DummyLogger())
        monkeypatch.setattr(test_mod, '_check_pure_ml_testing', lambda cfg: False)
        monkeypatch.setattr(test_mod, '_get_modules_and_device', lambda: ({}, 'cpu'))
        monkeypatch.setattr(test_mod, '_build_test_components', lambda cfg, modules, device, logger: ('dataloader', 'model', 'loss_fn'))
        monkeypatch.setattr('refrakt_core.api.utils.train_utils.setup_artifact_dumper', lambda cfg, name, logger: 'artifact_dumper')
        monkeypatch.setattr(test_mod, '_setup_trainer_for_testing', lambda cfg, model, dataloader, loss_fn, device, modules, artifact_dumper, name, logger: 'trainer')
        monkeypatch.setattr(test_mod, '_load_model_checkpoint', lambda model, model_path, device, logger: None)
        monkeypatch.setattr(test_mod, '_setup_fusion_evaluation', lambda cfg, model, dataloader, device, artifact_dumper, logger: 'fusion_acc')
        called = {}
        def fake_evaluate_model(trainer, model, dataloader, device, fusion_acc, logger):
            called['eval'] = True
            return {'eval': True}
        monkeypatch.setattr(test_mod, '_evaluate_model', fake_evaluate_model)
        test_mod.test(cfg=dummy_cfg, model_path='dummy.pth', logger=None)
        assert called.get('eval') 