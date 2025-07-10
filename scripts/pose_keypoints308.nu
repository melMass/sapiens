const root = (path self | path dirname -n 2)
const sapiens_checkpoint_root = "D:/AI-MODELS/Sapiens"

const model_map = [
  {
    name: goliath_best_goliath_AP_573
    size: "0.3b"
  }
  {
    name: goliath_best_goliath_AP_573
    size: "0.6b"
  }
  {
    name: goliath_best_goliath_AP_639
    size: "1b"
  }
]

const sizes = ["0.3b" "0.6b" "1b"]

# - timings for the full pipeline
# torchscript 7mn04sec using full graph and max_autotune - Compilation alone is much longer.
# bfloat16 1mn45 using NO full graph and default mode

def get-model [size: string] {
  $model_map | where size == $size | first
}

export def main [input_dir?: path output_dir?: path = "./output" --torchscript --size: string = "1b"] {

  if $size not-in $sizes {
    print $"(ansi red) Invalid size ($size). Valid sizes: ($sizes)"
    return
  }

  # activate the uv env
  overlay use ($root | path join .venv Scripts activate.nu)

  # fallback to demo
  let input_dir = ($input_dir | default { $root | path join pose demo data itw_videos reel1 })

  let mode = if $torchscript {
    'torchscript'
  } else {
    'bfloat16' ## 3xxx, A100 gpus. faster inference at bfloat16
  }

  print $"Using mode: ($mode)"

  let model = (get-model $size)
  let base_name = $'sapiens_($model.size)'
  let checkpoint = ($sapiens_checkpoint_root | path join $mode $"($base_name)_($model.name)_($mode).pt2")

  let output = ($output_dir | path join $base_name)

  mkdir $output

  let detection_config_file = (
    $root
    | path join pose demo mmdetection_cfg rtmdet_m_640-8xb32_coco-person_no_nms.py
  )

  let detection_checkpoint = (
    $sapiens_checkpoint_root
    | path join detector rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
  )

  #---------------------------VISUALIZATION PARAMS--------------------------------------------------
  let line_thickness = 3 ## line thickness of the skeleton
  let radius = 6 ## keypoint radius
  let kpt_thres = 0.3 ## confidence threshold

  ##-------------------------------------inference-------------------------------------
  let run_file = ($root | path join lite demo vis_pose.py)

  ## number of inference jobs per gpu, total number of gpus and gpu ids
  # JOBS_PER_GPU=1; TOTAL_GPUS=8; VALID_GPU_IDS=(0 1 2 3 4 5 6 7)
  let JOBS_PER_GPU = 1
  let total_gpus = 1
  let valid_gpu_ids = [0]

  let batch_size = 8

  let here = ($env.PWD)
  cd $input_dir
  let image_list = (glob "{*.jpg,*.png}" | sort | str replace -a '\' '/')

  let img_count = ($image_list | length)

  mut chunk_size = 0
  mut job_count = 0

  if (($total_gpus > $img_count // $batch_size)) {
    $job_count = (($img_count + $batch_size - 1) // $batch_size)
    $chunk_size = $batch_size
  } else {
    $job_count = ($JOBS_PER_GPU * $total_gpus)
    $chunk_size = ($img_count // $job_count)
  }

  $env.TF_CPP_MIN_LOG_LEVEL = 2

  print $"Distributing ($img_count) image paths into ($job_count) jobs."

  cd ($root | path join pose)

  $image_list | chunks $chunk_size | enumerate | par-each {|i|
    let text_file = ($input_dir | path join $"image_paths_($i.index + 1).txt")

    $i.item | str join "\n" | save -f $text_file

    let gpu_id = ($i.index mod $total_gpus)
    $env.CUDA_VISIBLE_DEVICES = ($valid_gpu_ids | get $gpu_id | default "0")
    python $run_file $checkpoint --num_keypoints 308 --det-config $detection_config_file --det-checkpoint $detection_checkpoint --batch-size $batch_size --input $text_file --output-root $output_dir --radius $radius --kpt-thr $kpt_thres --thickness $line_thickness ## add & to process in background
    # Allow a short delay between starting each job to reduce system load spikes
  }

  print "Processing complete."
  print $"Results saved to ($output_dir)"
}
