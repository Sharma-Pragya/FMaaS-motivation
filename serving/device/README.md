cd serving
python -m grpc_tools.protoc \
  -I . \
  --python_out=. \
  --grpc_python_out=. \
  device/proto/edge_runtime.proto

cd /nfs/obelix/users3/hshastri/FMaaS-motivation/serving && /nfs/obelix/users2/hshastri/anaconda3/envs/fmtk/bin/python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. device/proto/edge_runtime.proto 2>&1