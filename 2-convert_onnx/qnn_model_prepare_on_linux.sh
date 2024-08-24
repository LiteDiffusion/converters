#!/bin/bash

if [ "$1" = "" ]
then
    echo "Usage: $0 <path>"
    exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
STABLE_DIFFUSION_MODELS=$1

export QNN_SDK_ROOT="/opt/qcom/aistack/qnn/2.20.0.240223"
export PYTHONPATH=$QNN_SDK_ROOT/benchmarks/QNN:$QNN_SDK_ROOT/lib/python
export PATH=$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:$CONDA_PREFIX/lib
export HEXAGON_TOOLS_DIR=$QNN_SDK_ROOT/bin/x86_64-linux-clang

echo -------------------------------
echo Generate model inputs list/data
echo -------------------------------
inputs_pickle_path=$STABLE_DIFFUSION_MODELS/fp32.npy
python3 $SCRIPT_DIR/generate_inputs.py --pickle_path $inputs_pickle_path --working_dir $STABLE_DIFFUSION_MODELS

echo ------------------------
echo Convert the text encoder
echo ------------------------
mkdir -p $STABLE_DIFFUSION_MODELS/converted_text_encoder
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter \
    -o $STABLE_DIFFUSION_MODELS/converted_text_encoder/qnn_model.cpp \
    -i $STABLE_DIFFUSION_MODELS/text_encoder_onnx/text_encoder.onnx \
    --input_list $STABLE_DIFFUSION_MODELS/text_encoder_onnx/text_encoder_input_list.txt \
    --act_bw 16 \
    --bias_bw 8 \
    --quantization_overrides $STABLE_DIFFUSION_MODELS/text_encoder_onnx/text_encoder.encodings
mv $STABLE_DIFFUSION_MODELS/converted_text_encoder/qnn_model.cpp $STABLE_DIFFUSION_MODELS/converted_text_encoder/text_encoder.cpp
mv $STABLE_DIFFUSION_MODELS/converted_text_encoder/qnn_model.bin $STABLE_DIFFUSION_MODELS/converted_text_encoder/text_encoder.bin
mv $STABLE_DIFFUSION_MODELS/converted_text_encoder/qnn_model_net.json $STABLE_DIFFUSION_MODELS/converted_text_encoder/text_encoder_net.json

echo -------------
echo Convert U-Net
echo -------------
mkdir -p $STABLE_DIFFUSION_MODELS/converted_unet
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter \
    -o $STABLE_DIFFUSION_MODELS/converted_unet/qnn_model.cpp \
    -i $STABLE_DIFFUSION_MODELS/unet_onnx/unet.onnx \
    --input_list $STABLE_DIFFUSION_MODELS/unet_onnx/unet_input_list.txt \
    --act_bw 16 \
    --bias_bw 8 \
    --quantization_overrides $STABLE_DIFFUSION_MODELS/unet_onnx/unet.encodings
mv $STABLE_DIFFUSION_MODELS/converted_unet/qnn_model.cpp $STABLE_DIFFUSION_MODELS/converted_unet/unet.cpp
mv $STABLE_DIFFUSION_MODELS/converted_unet/qnn_model.bin $STABLE_DIFFUSION_MODELS/converted_unet/unet.bin
mv $STABLE_DIFFUSION_MODELS/converted_unet/qnn_model_net.json $STABLE_DIFFUSION_MODELS/converted_unet/unet_net.json

echo -----------------------
echo Convert the VAE decoder
echo -----------------------
mkdir -p $STABLE_DIFFUSION_MODELS/converted_vae_decoder
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter \
    -o $STABLE_DIFFUSION_MODELS/converted_vae_decoder/qnn_model.cpp \
    -i $STABLE_DIFFUSION_MODELS/vae_decoder_onnx/vae_decoder.onnx \
    --input_list $STABLE_DIFFUSION_MODELS/vae_decoder_onnx/vae_decoder_input_list.txt \
    --act_bw 16 \
    --bias_bw 8 \
    --quantization_overrides $STABLE_DIFFUSION_MODELS/vae_decoder_onnx/vae_decoder.encodings
mv $STABLE_DIFFUSION_MODELS/converted_vae_decoder/qnn_model.cpp $STABLE_DIFFUSION_MODELS/converted_vae_decoder/vae_decoder.cpp
mv $STABLE_DIFFUSION_MODELS/converted_vae_decoder/qnn_model.bin $STABLE_DIFFUSION_MODELS/converted_vae_decoder/vae_decoder.bin
mv $STABLE_DIFFUSION_MODELS/converted_vae_decoder/qnn_model_net.json $STABLE_DIFFUSION_MODELS/converted_vae_decoder/vae_decoder_net.json

echo ---------------------------------------
echo Generate the text encoder model library
echo ---------------------------------------
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator \
    -c $STABLE_DIFFUSION_MODELS/converted_text_encoder/text_encoder.cpp \
    -b $STABLE_DIFFUSION_MODELS/converted_text_encoder/text_encoder.bin \
    -t x86_64-linux-clang \
    -o $STABLE_DIFFUSION_MODELS/converted_text_encoder

echo --------------------------------
echo Generate the U-Net model library
echo --------------------------------
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator \
    -c $STABLE_DIFFUSION_MODELS/converted_unet/unet.cpp \
    -b $STABLE_DIFFUSION_MODELS/converted_unet/unet.bin \
    -t x86_64-linux-clang \
    -o $STABLE_DIFFUSION_MODELS/converted_unet

echo --------------------------------------
echo Generate the VAE decoder model library
echo --------------------------------------
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator \
    -c $STABLE_DIFFUSION_MODELS/converted_vae_decoder/vae_decoder.cpp \
    -b $STABLE_DIFFUSION_MODELS/converted_vae_decoder/vae_decoder.bin \
    -t x86_64-linux-clang \
    -o $STABLE_DIFFUSION_MODELS/converted_vae_decoder

cat > /tmp/htp_backend_extensions.json <<EOF
{
    "backend_extensions": {
        "shared_library_path": "libQnnHtpNetRunExtensions.so",
        "config_file_path": "/tmp/htp_config.json"
    }
}
EOF
cat > /tmp/htp_config.json <<EOF
{
    "graphs": {
        "vtcm_mb":8,
        "graph_names":["qnn_model"]
    },
    "devices": [
        {
            "dsp_arch": "v73",
            "cores":[{
                "core_id": 0,
                "perf_profile": "burst",
                "rpc_control_latency":100
            }]
        }
    ]
}
EOF
mkdir -p $STABLE_DIFFUSION_MODELS/serialized_binaries

echo ------------------------------------------------
echo Generate the QNN context binary for text encoder
echo ------------------------------------------------
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator \
    --model $STABLE_DIFFUSION_MODELS/converted_text_encoder/x86_64-linux-clang/libtext_encoder.so \
    --backend libQnnHtp.so \
    --output_dir $STABLE_DIFFUSION_MODELS/serialized_binaries \
    --binary_file text_encoder.serialized \
    --config_file /tmp/htp_backend_extensions.json

echo -----------------------------------------
echo Generate the QNN context binary for U-Net
echo -----------------------------------------
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator \
    --model $STABLE_DIFFUSION_MODELS/converted_unet/x86_64-linux-clang/libunet.so \
    --backend libQnnHtp.so \
    --output_dir $STABLE_DIFFUSION_MODELS/serialized_binaries \
    --binary_file unet.serialized \
    --config_file /tmp/htp_backend_extensions.json

echo -----------------------------------------------
echo Generate the QNN context binary for VAE Decoder
echo -----------------------------------------------
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator \
    --model $STABLE_DIFFUSION_MODELS/converted_vae_decoder/x86_64-linux-clang/libvae_decoder.so \
    --backend libQnnHtp.so \
    --output_dir $STABLE_DIFFUSION_MODELS/serialized_binaries \
    --binary_file vae_decoder.serialized \
    --config_file /tmp/htp_backend_extensions.json
