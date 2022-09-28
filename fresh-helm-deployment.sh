# First, delete the RayCluster custom resource.
kubectl -n ray delete raycluster example-cluster

# Delete the Ray release.
helm -n ray uninstall example-cluster

# Reinstall everything.
helm -n ray install example-cluster --create-namespace ./helm/hyadis-chart-tmp-data-volume/
