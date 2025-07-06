const root = (path self | path dirname -n 5)
const sapiens_checkpoint_root = "D:/AI-MODELS/Sapiens"

def model_complete [] {
}

export def main [input_dir?: path output_dir?: path = "./output" --torchscript] {
  print $root
  overlay use ($root | path join .venv Scripts activate.nu)

  let input_dir = ($input_dir | default { $root | path join pose demo data itw_videos reel1 })
  # SAPIENS_CHECKPOINT_ROOT=/home/${USER}/sapiens_lite_host

  # MODE='torchscript' ## original. no optimizations (slow). full precision inference.
  let mode = if $torchscript {
    'torchscript'
  } else {
    'bfloat16' ## A100 gpus. faster inference at bfloat16
  }

  # let SAPIENS_CHECKPOINT_ROOT=$SAPIENS_CHECKPOINT_ROOT/$MODE

  #----------------------------set your input and output directories----------------------------------------------
  # INPUT='../pose/demo/data/itw_videos/reel1'
  # OUTPUT="/home/${USER}/Desktop/sapiens/pose/Outputs/vis/itw_videos/reel1_pose308"

  #--------------------------MODEL CARD---------------
  # MODEL_NAME='sapiens_0.3b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/sapiens_0.3b/sapiens_0.3b_goliath_best_goliath_AP_573_$MODE.pt2
  # MODEL_NAME='sapiens_0.6b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/sapiens_0.6b/sapiens_0.6b_goliath_best_goliath_AP_609_$MODE.pt2
  # MODEL_NAME='sapiens_1b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_AP_639_$MODE.pt2

  let model_name = 'sapiens_1b'
  let checkpoint = ($sapiens_checkpoint_root | path join $"sapiens_1b_goliath_best_goliath_AP_639_($mode).pt2")

  let output = ($output_dir | path join $model_name)
  mkdir $output

  let detection_config_file = ($root | path join pose demo mmdetection_cfg rtmdet_m_640-8xb32_coco-person_no_nms.py)

  let detection_checkpoint = ($sapiens_checkpoint_root | path join detector rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth)

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

  # # Check if image list was created successfully
  # if [ ! -s "${IMAGE_LIST}" ]; then
  #   echo "No images found. Check your input directory and permissions."
  #   exit 1
  # fi

  # Count images and calculate the number of images per text file
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

  # Divide image paths into text files for each job
  # for ((i=0; i<TOTAL_JOBS; i++)); do
  #   TEXT_FILE="${INPUT}/image_paths_$((i+1)).txt"
  #   if [ $i -eq $((TOTAL_JOBS - 1)) ]; then
  #     # For the last text file, write all remaining image paths
  #     tail -n +$((IMAGES_PER_FILE * i + 1)) "${IMAGE_LIST}" > "${TEXT_FILE}"
  #   else
  #     # Write the exact number of image paths per text file
  #     head -n $((IMAGES_PER_FILE * (i + 1))) "${IMAGE_LIST}" | tail -n ${IMAGES_PER_FILE} > "${TEXT_FILE}"
  #   fi
  # done

  # for i in 0..($TOTAL_JOBS - 1) {
  #   let TEXT_FILE = ($input_dir | path join $"image_paths_($i + 1).txt")
  #   let start_idx = $i * $IMAGES_PER_FILE
  #   let end_idx = ($i + 1) * $IMAGES_PER_FILE - 1
  #   if $i == ($TOTAL_JOBS - 1) {
  #     $image_list | drop $start_idx | save $TEXT_FILE
  #   } else {
  #     $image_list | slice $start_idx..$end_idx | save $TEXT_FILE
  #   }
  # }

  $image_list | chunks $chunk_size | enumerate | par-each {|i|
    let text_file = ($input_dir | path join $"image_paths_($i.index + 1).txt")

    $i.item | str join "\n" | save -f $text_file

    let gpu_id = ($i.index mod $total_gpus)
    $env.CUDA_VISIBLE_DEVICES = ($valid_gpu_ids | get $gpu_id | default "0")
    python $run_file $checkpoint --num_keypoints 308 --det-config $detection_config_file --det-checkpoint $detection_checkpoint --batch-size $batch_size --input $text_file --output-root $output_dir --radius $radius --kpt-thr $kpt_thres --thickness $line_thickness ## add & to process in background
    # Allow a short delay between starting each job to reduce system load spikes
  }

  # Run the process on the GPUs, allowing multiple jobs per GPU
  # for ((i=0; i<TOTAL_JOBS; i++)); do

  # Remove the image list and temporary text files
  # rm "${IMAGE_LIST}"
  # for ((i=0; i<TOTAL_JOBS; i++)); do
  # rm "${INPUT}/image_paths_$((i+1)).txt"
  # done

  # Go back to the original script's directory
  # cd -

  print "Processing complete."
  print $"Results saved to ($output_dir)"
}
