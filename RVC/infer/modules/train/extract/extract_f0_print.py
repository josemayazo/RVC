import os
import sys
import traceback

import parselmouth

now_dir = os.getcwd()
sys.path.append(now_dir)
import logging

import numpy as np
import pyworld

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from infer.lib.audio import load_audio

logging.getLogger("numba").setLevel(logging.WARNING)
from multiprocessing import Process

# Global variables for log file
_log_file = None

def printt(strr):
    """Print and log function"""
    print(strr)
    if _log_file:
        _log_file.write("%s\n" % strr)
        _log_file.flush()


class FeatureInput(object):
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path, f0_method):
        x = load_audio(path, self.fs)
        p_len = x.shape[0] // self.hop
        if f0_method == "pm":
            time_step = 160 / 16000 * 1000
            f0_min = 50
            f0_max = 1100
            f0 = (
                parselmouth.Sound(x, self.fs)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "harvest":
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        elif f0_method == "dio":
            f0, t = pyworld.dio(
                x.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        elif f0_method == "rmvpe":
            if hasattr(self, "model_rmvpe") == False:
                from infer.lib.rmvpe import RMVPE

                print("Loading rmvpe model")
                self.model_rmvpe = RMVPE(
                    "assets/rmvpe/rmvpe.pt", is_half=False, device="cpu"
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        return f0

    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def go(self, paths, f0_method):
        if len(paths) == 0:
            printt("no-f0-todo")
        else:
            printt("todo-f0-%s" % len(paths))
            n = max(len(paths) // 5, 1)  # 每个进程最多打印5条
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    if idx % n == 0:
                        printt("f0ing,now-%s,all-%s,-%s" % (idx, len(paths), inp_path))
                    if (
                        os.path.exists(opt_path1 + ".npy") == True
                        and os.path.exists(opt_path2 + ".npy") == True
                    ):
                        continue
                    featur_pit = self.compute_f0(inp_path, f0_method)
                    np.save(
                        opt_path2,
                        featur_pit,
                        allow_pickle=False,
                    )  # nsf
                    coarse_pit = self.coarse_f0(featur_pit)
                    np.save(
                        opt_path1,
                        coarse_pit,
                        allow_pickle=False,
                    )  # ori
                except:
                    printt("f0fail-%s-%s-%s" % (idx, inp_path, traceback.format_exc()))


def extract_f0_print(exp_dir, n_p=2, f0method="pm"):
    """
    Main function to extract F0 features
    
    Args:
        exp_dir: Experiment directory path
        n_p: Number of processes to use
        f0method: F0 extraction method ('pm', 'harvest', 'dio', 'rmvpe')
    """
    global _log_file
    
    # Setup logging
    log_path = "%s/extract_f0_feature.log" % exp_dir
    _log_file = open(log_path, "a+")
    
    try:
        printt(f"extract_f0_print: exp_dir={exp_dir}, n_p={n_p}, f0method={f0method}")
        
        featureInput = FeatureInput()
        paths = []
        inp_root = "%s/1_16k_wavs" % (exp_dir)
        opt_root1 = "%s/2a_f0" % (exp_dir)
        opt_root2 = "%s/2b-f0nsf" % (exp_dir)

        # Check if input directory exists
        if not os.path.exists(inp_root):
            printt(f"Input directory not found: {inp_root}")
            return False

        os.makedirs(opt_root1, exist_ok=True)
        os.makedirs(opt_root2, exist_ok=True)
        
        for name in sorted(list(os.listdir(inp_root))):
            inp_path = "%s/%s" % (inp_root, name)
            if "spec" in inp_path:
                continue
            opt_path1 = "%s/%s" % (opt_root1, name)
            opt_path2 = "%s/%s" % (opt_root2, name)
            paths.append([inp_path, opt_path1, opt_path2])

        if not paths:
            printt("No audio files found for processing")
            return False

        # Process files
        ps = []
        for i in range(n_p):
            p = Process(
                target=featureInput.go,
                args=(
                    paths[i::n_p],
                    f0method,
                ),
            )
            ps.append(p)
            p.start()
        
        for i in range(n_p):
            ps[i].join()
            
        printt("F0 extraction completed successfully")
        return True
        
    except Exception as e:
        printt(f"Error in F0 extraction: {str(e)}")
        printt(traceback.format_exc())
        return False
    finally:
        if _log_file:
            _log_file.close()
            _log_file = None


if __name__ == "__main__":
    # Command line interface for backward compatibility
    if len(sys.argv) < 4:
        print("Usage: python extract_f0_print.py <exp_dir> <n_p> <f0method>")
        sys.exit(1)
        
    exp_dir = sys.argv[1]
    n_p = int(sys.argv[2])
    f0method = sys.argv[3]
    
    extract_f0_print(exp_dir, n_p, f0method)
