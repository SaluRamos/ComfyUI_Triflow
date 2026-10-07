"""Private Windows inference entry point; no ComfyUI imports or WSL."""
import json
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    request = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    os.environ['TRIFLOW_DEVICE'] = request.get('device', 'cuda')
    if request.get('device', 'cuda').split(':')[0] == 'cuda':
        os.environ['TRIFLOW_VAE_SPARSE_BACKEND'] = 'spconv'
        os.environ['TRIFLOW_FLOW_SPARSE_BACKEND'] = 'spconv'
        os.environ['SPCONV_ALGO'] = 'native'
        os.environ['ATTN_BACKEND'] = 'flash_attn'
        os.environ['SPARSE_ATTN_BACKEND'] = 'flash_attn'
    import inference
    runtime = inference.load_inference_runtime(request['checkpoints'])
    outputs = inference.run_inference(str(Path(request['inputs'][0]).parent), request['output'],
        rank=0, world_size=1, face_count=request['face_count'],
        qem_threshold=request['qem_threshold'], quad_ratio=request['quad_ratio'],
        runtime=runtime, mesh_paths=request['inputs'])
    Path(request['response']).write_text(json.dumps([str(path) for path in outputs]), encoding='utf-8')
