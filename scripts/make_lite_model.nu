use log.nu

const root = (path self | path dirname -n 2)

export def main [model: path --torchscript] {

  let mode = if $torchscript { "torchscript" } else { "bfloat16" }

  let parsed = ($model | path basename | parse "sapiens_{size}_{dataset}_{_rest}.{ext}")
  if ($parsed | is-empty) {
    log error "Could not parse model name"
    return
  }
  let parsed = ($parsed | first)
  let model_name = $"sapiens_($parsed.size)"
  let out_name = $"($model_name)_($parsed.dataset)-1024x768"
  let get_config = {||
    cd ($root | path join pose configs sapiens_pose $parsed.dataset)
    try { glob $"sapiens_($parsed.size)*" } | default [null] | first
  }

  let config = (do $get_config)
  if ($config | is-empty) {
    log error $"Could not find config for ($parsed.dataset)"
    return
  }

  $env.TORCHDYNAMO_VERBOSE = 1
  $env.CUDA_VISIBLE_DEVICES = 0
  python ($root | path join pose tools deployment torch_optimization.py) $config $model --explain-verbose --output-dir "./output"
}
