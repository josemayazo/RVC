import os
import sys
import traceback

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

# Global variables for logging
_log_file = None

def printt(strr):
    """Print and log function"""
    print(strr)
    if _log_file:
        _log_file.write("%s\n" % strr)
        _log_file.flush()

def setup_device(device_arg=None):
    """Setup the computation device"""
    if device_arg and "privateuseone" in device_arg:
        import torch_directml
        device = torch_directml.device(torch_directml.default_device())
        
        def forward_dml(ctx, x, scale):
            ctx.scale = scale
            res = x.clone().detach()
            return res
        
        fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml
        return device
    else:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        return device
def extract_feature_print(device_arg="cuda:0", n_part=1, i_part=0, exp_dir=None, version="v2", is_half=True, i_gpu=None):
    """
    Main function to extract HuBERT features
    
    Args:
        device_arg: Device to use for computation ('cpu', 'cuda:0', etc.)
        n_part: Number of parts to divide the work
        i_part: Which part this process handles
        exp_dir: Experiment directory path
        version: Model version ('v1' or 'v2')
        is_half: Whether to use half precision
        i_gpu: GPU index (optional)
    """
    global _log_file
    
    if exp_dir is None:
        raise ValueError("exp_dir must be provided")
    
    # Setup device
    device = setup_device(device_arg)
    if i_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    
    # Setup logging
    log_path = "%s/extract_f0_feature.log" % exp_dir
    _log_file = open(log_path, "a+")
    
    try:
        printt(f"extract_feature_print: device={device}, exp_dir={exp_dir}, version={version}")

        model_path = "assets/hubert/hubert_base.pt"

        printt("exp_dir: " + exp_dir)
        wavPath = "%s/1_16k_wavs" % exp_dir
        outPath = (
            "%s/3_feature256" % exp_dir if version == "v1" else "%s/3_feature768" % exp_dir
        )
        os.makedirs(outPath, exist_ok=True)

        # Check if input directory exists
        if not os.path.exists(wavPath):
            printt(f"Input directory not found: {wavPath}")
            return False

        # wave must be 16k, hop_size=320
        def readwave(wav_path, normalize=False):
            wav, sr = sf.read(wav_path)
            assert sr == 16000
            feats = torch.from_numpy(wav).float()
            if feats.dim() == 2:  # double channels
                feats = feats.mean(-1)
            assert feats.dim() == 1, feats.dim()
            if normalize:
                with torch.no_grad():
                    feats = F.layer_norm(feats, feats.shape)
            feats = feats.view(1, -1)
            return feats

        # HuBERT model
        printt("load model(s) from {}".format(model_path))
        # if hubert model is exist
        if os.access(model_path, os.F_OK) == False:
            printt(
                "Error: Extracting is shut down because %s does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
                % model_path
            )
            return False
            
        models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [model_path],
            suffix="",
        )
        model = models[0]
        model = model.to(device)
        printt("move model to %s" % device)
        if is_half:
            if device not in ["mps", "cpu"]:
                model = model.half()
        model.eval()

        todo = sorted(list(os.listdir(wavPath)))[i_part::n_part]
        n = max(1, len(todo) // 10)  # 最多打印十条
        if len(todo) == 0:
            printt("no-feature-todo")
            return True
        else:
            printt("all-feature-%s" % len(todo))
            for idx, file in enumerate(todo):
                try:
                    if file.endswith(".wav"):
                        wav_path = "%s/%s" % (wavPath, file)
                        out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))

                        if os.path.exists(out_path):
                            continue

                        feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
                        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                        inputs = {
                            "source": (
                                feats.half().to(device)
                                if is_half and device not in ["mps", "cpu"]
                                else feats.to(device)
                            ),
                            "padding_mask": padding_mask.to(device),
                            "output_layer": 9 if version == "v1" else 12,  # layer 9
                        }
                        with torch.no_grad():
                            logits = model.extract_features(**inputs)
                            feats = (
                                model.final_proj(logits[0]) if version == "v1" else logits[0]
                            )

                        feats = feats.squeeze(0).float().cpu().numpy()
                        if np.isnan(feats).sum() == 0:
                            np.save(out_path, feats, allow_pickle=False)
                        else:
                            printt("%s-contains nan" % file)
                        if idx % n == 0:
                            printt("now-%s,all-%s,%s,%s" % (len(todo), idx, file, feats.shape))
                except:
                    printt(traceback.format_exc())
            printt("all-feature-done")
            return True
            
    except Exception as e:
        printt(f"Error in feature extraction: {str(e)}")
        printt(traceback.format_exc())
        return False
    finally:
        if _log_file:
            _log_file.close()
            _log_file = None


if __name__ == "__main__":
    # Command line interface for backward compatibility
    if len(sys.argv) < 4:
        print("Usage: python extract_feature_print.py <device> <n_part> <i_part> [i_gpu] <exp_dir> <version> <is_half>")
        sys.exit(1)
    
    device = sys.argv[1]
    n_part = int(sys.argv[2])
    i_part = int(sys.argv[3])
    
    if len(sys.argv) == 7:
        exp_dir = sys.argv[4]
        version = sys.argv[5]
        is_half = sys.argv[6].lower() == "true"
        extract_feature_print(device, n_part, i_part, exp_dir, version, is_half)
    else:
        i_gpu = sys.argv[4]
        exp_dir = sys.argv[5]
        version = sys.argv[6]
        is_half = sys.argv[7].lower() == "true"
        extract_feature_print(device, n_part, i_part, exp_dir, version, is_half, i_gpu)
