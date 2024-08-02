
# Script to build and install all benchmark dependencies
# Change the dirs in Setting ENV vars for your local setup
echo "===BM===Setting ENV vars for OneAPI==="
# compiler
export LDIR=/localdisk/jojimonv/one_api_ver596
source ${LDIR}/setvars.sh

# oneMKL
export MKL_DPCPP_ROOT=${LDIR}/mkl/latest/
export LD_LIBRARY_PATH=${MKL_DPCPP_ROOT}/lib:${MKL_DPCPP_ROOT}/lib64:${MKL_DPCPP_ROOT}/lib/intel64:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${MKL_DPCPP_ROOT}/lib:${MKL_DPCPP_ROOT}/lib64:${MKL_DPCPP_ROOT}/lib/intel64:$LIBRARY_PATH

#export USE_AOT_DEVLIST='pvc'
export USE_AOT_DEVLIST='ats-m150,pvc'

# export BUILD_SEPARATE_OPS=ON
export BUILD_WITH_CPU=OFF
export USE_XETLA=OFF
export USE_XPU=1

echo "===BM======Basic requirements========"
pip install numpy transformers scikit-learn pyproject-toml 
pip install libpng-bins pylibjpeg-libjpeg ffmpeg
pip install backports.lzma
pip install "git+https://github.com/huggingface/pytorch-image-models@730b907b4d45a4713cbc425cbf224c46089fd514" --no-deps

#
conda install libpng jpeg
conda install -c conda-forge ffmpeg
#conda install git-lfs pyyaml pandas scipy psutil
#
#
echo "===BM======PYTORCH=============Pytorch 2.3"
git clone https://github.com/intel-innersource/frameworks.ai.pytorch.private-gpu
git apply --directory=frameworks.ai.pytorch.private-gpu pytorch.dll.patch
pushd frameworks.ai.pytorch.private-gpu
git status
git submodule sync 
git submodule update --init --recursive
pip install -r requirements.txt
python setup.py develop
python setup.py install
export torch_src_folder=`pwd`
popd
echo " TORCH DIR: ${torch_src_folder}"


echo "===BM=====Create Tem build dir========"
mkdir build
pushd build
echo "===BM======IPEX================"
git clone https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-gpu
pushd frameworks.ai.pytorch.ipex-gpu
pip install -r requirements.txt
git submodule sync 
git submodule update --init --recursive --jobs 0
python setup.py develop
python setup.py install
popd

echo "===BM======Triton================"
wget https://github.com/intel/intel-xpu-backend-for-triton/releases/download/v3.0.0b2/triton_xpu-3.0.0b2-cp39-cp39-linux_x86_64.whl
pip install triton_xpu-3.0.0b2-cp39-cp39-linux_x86_64.whl


echo "===BM======Vision============="
git clone https://github.com/pytorch/vision.git
pushd vision
#pip install -r requirements.txt
git submodule sync 
git submodule update --init --recursive
git checkout `cat ${torch_src_folder}/.github/ci_commit_pins/vision.txt`
python setup.py clean install
popd

echo "===BM======Text============="
git clone https://github.com/pytorch/text
pushd text
pip install -r requirements.txt
git submodule sync 
git submodule update --init --recursive
git checkout `cat ${torch_src_folder}/.github/ci_commit_pins/text.txt`
python setup.py clean install
popd


echo "===BM======Audio============="
git clone https://github.com/pytorch/audio.git
pushd audio
pip install -r requirements.txt
git submodule sync 
git submodule update --init --recursive
git checkout `cat ${torch_src_folder}/.github/ci_commit_pins/audio.txt`
python setup.py clean install
popd

echo "=============================================================="
echo "Testing versions of Torch, Vision, Text, Audio"
echo "=============================================================="
python -c "import torchvision,torchtext,torchaudio;print(torchvision.__version__, torchtext.__version__, torchaudio.__version__)"
echo "=============================================================="

echo "===BM======Temp benchmarks===="
pip install git-lfs pyyaml pandas psutil pyre_extensions torchrec 
git clone https://github.com/weishi-deng/benchmark
pushd benchmark
pip install -r requirements.txt
python install.py
# Note that -e is necessary
pip install -e .
popd
popd
cd frameworks.ai.pytorch.private-gpu
echo "===BM======Run Sample===="
bash ../scripts/inductor_test.sh huggingface amp_fp16 training performance xpu 0 static 1 0 T5Small
#echo "===BM======Deleteing repos===="
#\rm -rf build
echo "===BM======Done...!===="

