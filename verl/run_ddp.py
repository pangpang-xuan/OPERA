# encoding=utf-8
import os
import socket
import json
import argparse
import subprocess
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

def get_params_from_mlp_config():
    cluster_spec = json.loads(os.environ["AFO_ENV_CLUSTER_SPEC"])
    print(cluster_spec)
    role = cluster_spec["role"]
    assert role == "worker", "{} vs worker".format(role)
    node_rank = int(cluster_spec["index"])
    nnodes = len(cluster_spec[role])
    master = cluster_spec[role][0]
    print(master)
    master_addr, master_ports = master.split(":")
    master_addr = socket.gethostbyname(master_addr)
    master_ports = master_ports.split(",")
    master_port = master_ports[0]
    return nnodes, node_rank, master_addr, master_port

if __name__ == '__main__':
    # command = sys.argv[1]
    nnodes, node_rank, master_addr, master_port = get_params_from_mlp_config()
    print(f"gloabl config ddp arguments\nnnodes={nnodes}, node_rank={node_rank}, master_addr={master_addr}, master_port={master_port}")
    # distritued_args = f"--nproc_per_node=auto --nnodes={nnodes} --node_rank={node_rank} --master_addr={master_addr} --master_port={master_port}"
    # train_cmd = [f"CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun {distritued_args} {command}"]
    train_cmd = [f"sh examples/sft/gsm8k/run_qwen_05_sp2.sh {nnodes} {node_rank} {master_addr} {master_port}"]
    print(train_cmd)
    result = subprocess.check_call(train_cmd, shell=True)