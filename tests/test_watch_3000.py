"""Recovery mechanics use temporary files and never spawn a GPU process."""
from pathlib import Path

import pytest

from experiments.teacher_context_grpo import watch_3000 as watcher


def fixture_run(tmp_path,monkeypatch,error='Teacher API HTTP 429'):
    root=tmp_path/'run';execution=root/'resume_3000_429_20261112';monitoring=execution/'monitoring'
    monitoring.mkdir(parents=True);(root/'updates').mkdir();checkpoint=root/'checkpoints/step_01000';checkpoint.mkdir(parents=True)
    (checkpoint/'state.pt').write_bytes(b'fixture checkpoint bytes')
    watcher.write(checkpoint/'metadata.json',{'step':1000,'state_sha256':watcher.sha(checkpoint/'state.pt')})
    watcher.write(root/'checkpoints/latest.json',{'step':1000,'path':'step_01000'})
    watcher.write(root/'process.json',{'worker_pid':11,'supervisor_pid':22})
    watcher.write(root/'worker_error.json',{'type':'InfrastructureError','error':error})
    watcher.write(root/'status.json',{'phase':'needs_attention','completed':1036})
    watcher.write(root/'teacher_api_usage.json',{'requests':163,'retries':61})
    (root/'updates/step_01000.json.gz').write_bytes(b'committed')
    (root/'updates/step_01001.json.gz').write_bytes(b'unsaved')
    (execution/'worker.log').write_text('original failure evidence')
    monkeypatch.setattr(watcher,'verify_execution',lambda *a:None)
    monkeypatch.setattr(watcher,'owned_alive',lambda *a:False)
    return root,execution,monitoring


def test_recovery_archives_unsaved_suffix_and_keeps_api_accounting(tmp_path,monkeypatch):
    root,execution,monitoring=fixture_run(tmp_path,monkeypatch)
    event=watcher.prepare_recovery(root,execution,monitoring,1)
    assert event['resumed_from_step']==1000 and len(event['archived_uncheckpointed_updates'])==1
    assert (root/'updates/step_01000.json.gz').read_bytes()==b'committed'
    assert not (root/'updates/step_01001.json.gz').exists()
    history=monitoring/'recoveries/attempt_01'
    assert (history/'uncheckpointed_updates/step_01001.json.gz').read_bytes()==b'unsaved'
    assert (history/'execution_worker.log').read_text()=='original failure evidence'
    assert watcher.read(root/'teacher_api_usage.json')=={'requests':163,'retries':61}
    assert watcher.read(root/'status.json')['expected']==3000


def test_live_worker_blocks_recovery_without_mutation(tmp_path,monkeypatch):
    root,execution,monitoring=fixture_run(tmp_path,monkeypatch)
    monkeypatch.setattr(watcher,'owned_alive',lambda *a:True)
    with pytest.raises(RuntimeError,match='still alive'):
        watcher.prepare_recovery(root,execution,monitoring,1)
    assert (root/'process.json').exists() and not (monitoring/'recoveries').exists()


def test_permanent_error_and_exhausted_budget_do_not_restart(tmp_path,monkeypatch):
    root,execution,monitoring=fixture_run(tmp_path,monkeypatch,error='Teacher API HTTP 401')
    with pytest.raises(RuntimeError,match='not eligible'):
        watcher.prepare_recovery(root,execution,monitoring,1)
    assert not watcher.eligible_error({'type':'ContractError','error':'Teacher API HTTP 429'})
    watcher.write(root/'worker_error.json',{'type':'InfrastructureError','error':'Teacher API HTTP 429'})
    watcher.write(monitoring/'config.json',{'max_automatic_restarts':0,'stall_seconds':2100,'poll_seconds':30})
    monkeypatch.setattr(watcher.subprocess,'Popen',lambda *a,**kw:pytest.fail('Unexpected launch'))
    watcher.monitor(root,execution,monitoring)
    assert watcher.read(monitoring/'status.json')['reason']=='Automatic restart budget exhausted'


def test_hourly_timer_persists_cadence_without_duplicate_or_fabricated_checks(tmp_path,monkeypatch):
    root,execution,monitoring=fixture_run(tmp_path,monkeypatch)
    monkeypatch.setattr(watcher,'gpu_snapshot',lambda:{'available':True,'index':2})
    config={'hourly_audit_seconds':3600};snapshot={'phase':'training','completed':1000}
    assert watcher.hourly_audit_if_due(root,execution,monitoring,snapshot,config,now=10000)
    assert not watcher.hourly_audit_if_due(root,execution,monitoring,snapshot,config,now=13599)
    assert watcher.hourly_audit_if_due(root,execution,monitoring,snapshot,config,now=13600)
    assert len(list((monitoring/'hourly_reports').glob('*.json')))==2
    assert watcher.read(monitoring/'hourly_schedule.json')['next_check_epoch']==17200
    # After a delayed supervisor restart, write one actual observation and keep
    # the cadence rather than inventing results for each missed interval.
    assert watcher.hourly_audit_if_due(root,execution,monitoring,snapshot,config,now=21000)
    assert watcher.read(monitoring/'hourly_schedule.json')['next_check_epoch']==24400
    assert len(list((monitoring/'hourly_reports').glob('*.json')))==3
