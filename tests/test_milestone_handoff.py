"""Check the milestone handoff without signalling real processes or using GPUs."""
import signal
from pathlib import Path

import pytest

from experiments.teacher_context_grpo import finish_at_milestone as controller


def prepared(tmp_path, target=1000):
    root=tmp_path/'run';out=root/f'eval_after_{target}';out.mkdir(parents=True)
    (root/'checkpoints').mkdir();(root/'source').mkdir()
    controller.write(root/'receipt.json',{'unchanged':True})
    identities={'worker':{'pid':111},'supervisor':{'pid':222}}
    config={'target':target,'original_receipt_sha256':controller.sha(root/'receipt.json'),
            'controller_sha256':controller.sha(Path(controller.__file__)),**identities}
    controller.write(out/'execution_receipt.json',config)
    return root,out,identities


def test_saved_milestone_stops_only_owned_processes_then_evaluates(tmp_path,monkeypatch):
    root,out,identities=prepared(tmp_path)
    checkpoint=root/'checkpoints/step_01000';checkpoint.mkdir()
    controller.write(checkpoint/'metadata.json',{'step':1000})
    controller.write(root/'checkpoints/latest.json',{'step':1000})
    alive={111,222};signals=[];launches=[]
    monkeypatch.setattr(controller,'still_owned',lambda identity,root:identity['pid'] in alive)
    def kill(pid,sig):
        assert pid in {111,222}
        signals.append((pid,sig))
        if sig==signal.SIGTERM:alive.discard(pid)
    monkeypatch.setattr(controller.os,'kill',kill)
    class Evaluation:
        pid=333
        def wait(self):
            controller.write(out/'summary.json',{'complete':True,'completed':1000})
            return 0
    def popen(command,**kwargs):
        assert not alive
        assert command[command.index('--target')+1]=='1000' and '--evaluate' in command
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='2'
        launches.append(command)
        return Evaluation()
    monkeypatch.setattr(controller.subprocess,'Popen',popen)
    controller.supervise(root,out,1000)
    assert signals==[(111,signal.SIGSTOP),(222,signal.SIGTERM),(111,signal.SIGTERM),(111,signal.SIGCONT)]
    assert len(launches)==1 and controller.read(out/'cleanup.json')['own_eval_process_ended']
    assert controller.read(root/'receipt.json')=={'unchanged':True}


def test_failed_original_run_never_launches_evaluation_or_signals_processes(tmp_path,monkeypatch):
    root,out,_=prepared(tmp_path)
    controller.write(root/'checkpoints/latest.json',{'step':200})
    controller.write(root/'status.json',{'phase':'needs_attention'})
    monkeypatch.setattr(controller.os,'kill',lambda *a:pytest.fail('Unexpected signal'))
    monkeypatch.setattr(controller.subprocess,'Popen',lambda *a,**kw:pytest.fail('Unexpected evaluation'))
    with pytest.raises(RuntimeError,match='stopped before requested milestone'):
        controller.supervise(root,out,1000)
