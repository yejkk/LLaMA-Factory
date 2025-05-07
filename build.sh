BASE_IMAGE=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-cpu-py312-ubuntu22.04-ec2
PIP_INDEX="https://pypi.org/simple"


docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg PIP_INDEX="${PIP_INDEX}" \
    -t ${inference_image}:${VERSION} .