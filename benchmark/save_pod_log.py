import time
from kubernetes import client, config
import kubernetes

config.load_kube_config()
v1 = client.CoreV1Api()

while True:
    ret = v1.list_namespaced_pod(namespace='default')
    for item in ret.items:
        pod_name = item.metadata.name
        if 'images' in pod_name or 'monitor' in pod_name or 'pod' in pod_name:
            pass
        else:
            try:
                log = v1.read_namespaced_pod_log(name=pod_name, namespace='default')
                with open(f'./pod_logger/{pod_name}.txt', 'w') as f:
                    f.writelines(log)
            except kubernetes.client.exceptions.ApiException:
                pass
    time.sleep(5)

