"""Only terminate a service started by this context; never reuse/stop port 5084."""
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import requests
from urllib.parse import urlparse
from .contracts import ContractError,InfrastructureError
from .io import atomic_json


@contextmanager
def own_service(config,output,enabled=False):
    service=config["environment"]
    if not enabled:
        yield
        return
    parsed=urlparse(service["base_url"])
    if parsed.hostname!="127.0.0.1" or parsed.port==5084:
        raise ContractError("Managed SCOPE service must use its own localhost port")
    sock=socket.socket()
    try:
        if sock.connect_ex((parsed.hostname,parsed.port))==0:
            raise InfrastructureError("Requested service port is already occupied; no existing process was stopped")
    finally:sock.close()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    args=[service["python"],"-m","scope.service","--shop-root",service["shop_root"],
          "--longagent-root",service["longagent_root"],"--manifest",service["manifest"],
          "--train-products",service["train_products"],"--eval-products",service["eval_products"],
          "--port",str(parsed.port),"--owner-pid",str(os.getpid())]
    env=dict(os.environ);env["CUDA_VISIBLE_DEVICES"]="";env["JAVA_HOME"]=service["java_home"]
    env["PATH"]=env["JAVA_HOME"]+"/bin:"+env.get("PATH","")
    env["PYTHONPATH"]=str(Path(__file__).resolve().parents[1])+os.pathsep+env.get("PYTHONPATH","")
    with (output/"service.log").open("a",encoding="utf-8") as log:
        child=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT)
        atomic_json(output/"service_process.json",{"pid":child.pid,"port":parsed.port,"command":args})
        try:
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                if child.poll() is not None:raise InfrastructureError("SCOPE service exited during initialization; inspect service.log")
                try:
                    if requests.get(service["base_url"]+"/healthz",timeout=2).json().get("ready"):break
                except (requests.RequestException,ValueError):pass
                time.sleep(1)
            else:raise InfrastructureError("SCOPE service initialization timed out")
            yield
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=15)
            atomic_json(output/"cleanup.json",{"own_processes_ended":child.poll() is not None,"service_pid":child.pid,
                        "port":parsed.port,"original_5084_untouched":True})
