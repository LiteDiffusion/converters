# ==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright 2023 Qualcomm Technologies, Inc. All rights reserved.
#  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
#
#  The party receiving this software directly from QTI (the "Recipient")
#  may use this software as reasonably necessary solely for the purposes
#  set forth in the agreement between the Recipient and QTI (the
#  "Agreement"). The software may be used in source code form solely by
#  the Recipient's employees (if any) authorized by the Agreement. Unless
#  expressly authorized in the Agreement, the Recipient may not sublicense,
#  assign, transfer or otherwise provide the source code to any third
#  party. Qualcomm Technologies, Inc. retains all ownership rights in and
#  to the software
#
#  This notice supersedes any other QTI notices contained within the software
#  except copyright notices indicating different years of publication for
#  different portions of the software. This notice does not supersede the
#  application of any third party copyright notice to that third party's
#  code.
#
#  @@-COPYRIGHT-END-@@
# ==============================================================================


import numpy as np 
import argparse, os

# create sub-directory per model
def create_directory(name):
    cur_dir = model_dir + "/" + name
    if not os.path.exists(cur_dir): os.mkdir(cur_dir)
    cur_input_dir = cur_dir + "/" + name + "_inputs"
    if not os.path.exists(cur_input_dir): os.mkdir(cur_input_dir)
    return cur_dir, cur_input_dir

# load text encoder inputs
def load_te_inputs(src):
    te_dir, te_input_dir = create_directory("text_encoder_onnx")
    
    te_inputs = src[0]["token_ids"].astype(np.float32)
    with open(te_dir + "/text_encoder_input_list.txt", "w") as f:
        for i in range(len(te_inputs)):
            entry_name = f"{te_input_dir}/te_input_{i+1}.bin"
            te_inputs[i].tofile(entry_name)
            # write to input_list.txt
            f.write(entry_name + "\n")       

# load U-Net inputs
def load_unet_inputs(src):
    unet_dir, unet_input_dir = create_directory("unet_onnx")

    with open(unet_dir + "/unet_input_list.txt", "w") as f:
        for i, (latent, time_emb, hidden) in enumerate(src[0]["unet_inputs"]):
            latent_entry = f"{unet_input_dir}/unet_input_latent_{i+1}_1.bin"
            latent.transpose(0,2,3,1).tofile(latent_entry)
            timm_emb_entry = f"{unet_input_dir}/unet_input_time_embedding_{i+1}_2.bin"
            time_emb.tofile(timm_emb_entry)
            hidden_entry = f"{unet_input_dir}/unet_input_hidden_{i+1}_3.bin"
            hidden.tofile(hidden_entry)
            # Write to input_list.txt; ensure 3 inputs in one line
            f.write(latent_entry + " " + timm_emb_entry + " " + hidden_entry + "\n")       

# Load VAE inputs
def load_vae_inputs(src):
    vae_dir, vae_input_dir = create_directory("vae_decoder_onnx")

    vae_iputs = src[0]["latent"]
    with open(vae_dir + "/vae_decoder_input_list.txt", "w") as f:
        vae_iputs.transpose(0,2,3,1).tofile(f"{vae_input_dir}/vae_input_1.bin")
        # write to input_list.txt
        f.write(f"{vae_input_dir}/vae_input_1.bin")

def setup_inputs_for_qnn_converter():
    # load numpy array object
    np_object = np.load(pickle_path, allow_pickle=True)

    print("Load text encoder inputs data")
    load_te_inputs(np_object)
    print("Load U-Net inputs data")
    load_unet_inputs(np_object)
    print("Load VAE decoder inputs data")
    load_vae_inputs(np_object)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pp", "--pickle_path", help="PATH to numpy array object pickle generated in AIMET work-flow", required=True) 
    parser.add_argument("-wd", "--working_dir", help="PATH for creating qnn_assets directory", required=True, default="./qnn_assets")

    args = parser.parse_args()

    # path to numpy array object pickle generated in AIMET work-flow
    pickle_path = args.pickle_path

    # Create qnn_assets directory
    working_dir = args.working_dir 
    if not os.path.exists(working_dir): os.mkdir(working_dir)

    model_dir = working_dir

    setup_inputs_for_qnn_converter()