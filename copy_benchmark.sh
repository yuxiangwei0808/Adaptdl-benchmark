
kubectl cp benchmark-hyadis/ ray/$RAY_HEAD_POD_ID:benchmark-hyadis

kubectl -n ray exec -it $RAY_HEAD_POD_ID -- /bin/bash
