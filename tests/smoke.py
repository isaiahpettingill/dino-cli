"""Real OpenVINO CPU inference; Python is a development-only fixture generator.
Run: uv run --with openvino --with pillow python tests/smoke.py
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import openvino as ov
from openvino import opset13 as ops
from PIL import Image

root = Path(__file__).resolve().parents[1]
binary = root / 'target/debug/dino-cli'
env = os.environ.copy()
libs = str(Path(ov.__file__).parent / 'libs')
env['LD_LIBRARY_PATH'] = libs + ':' + env.get('LD_LIBRARY_PATH', '')
library = str(next(Path(libs).glob('libopenvino_c.so*')))
with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)
    x = ops.parameter([1, 3, 2, 2], ov.Type.f32, name='pixel_values')
    y = ops.reduce_mean(x, [2, 3], False)
    y.output(0).get_tensor().set_names({'embedding'})
    ov.save_model(ov.Model([y], [x]), d / 'openvino_model.xml', compress_to_fp16=False)
    (d / 'preprocessor_config.json').write_text(json.dumps({'do_resize': True, 'size': {'width': 2, 'height': 2}, 'resample': 0, 'do_normalize': False, 'do_rescale': True, 'rescale_factor': 1/255}))
    Image.new('RGB', (4, 4), (255, 0, 0)).save(d / 'red.png')
    Image.new('RGB', (4, 4), (0, 255, 0)).save(d / 'green.png')
    def run(*args, ok=True):
        p = subprocess.run([str(binary), '--ov-library', library, '-m', str(d), '--device', 'CPU', *map(str,args)], env=env, capture_output=True, text=True)
        assert (p.returncode == 0) == ok, (args, p.stdout, p.stderr)
        return json.loads(p.stdout) if ok else p.stderr
    assert run('embed', d/'red.png')['embedding'] == [1, 0, 0]
    assert abs(run('--json','compare',d/'red.png',d/'red.png')['similarity']-1) < 1e-6
    assert run('--json','compare',d/'red.png',d/'green.png')['similarity'] == 0
    assert run('--json','nearest',d/'red.png',d,'-k','1')[0]['image'].endswith('red.png')
    assert run('infer',d/'red.png')[0]['shape'] == [1,3]
    assert run('classify',d/'red.png','-k','1')[0]['index'] == 0
    assert len(run('info')['inputs']) == 1
    out = d/'vector.f32'
    subprocess.run([str(binary),'--ov-library',library,'-m',str(d),'--device','CPU','embed',str(d/'red.png'),'--output',str(out)], env=env, check=True)
    assert out.stat().st_size == 12
    assert 'Unknown output' in run('--output-name','missing','embed',d/'red.png',ok=False)
    assert 'positive' in run('nearest',d/'red.png',d,'-k','0',ok=False)
    run('--size','2x2','embed',d/'red.png')
    assert run('--stretch','--mean','0,0,0','--std','1,1,1','--scale','0.003921568627','embed',d/'red.png')['embedding'] == [1,0,0]
    batch = d/'batch.jsonl'
    subprocess.run([str(binary),'--ov-library',library,'-m',str(d),'--device','CPU','batch',str(d),'--output',str(batch)],env=env,check=True)
    assert len(batch.read_text().splitlines()) == 2
    # Offline Hugging Face cache: resolves the exact same IR/weights without networking.
    import shutil
    hub = d/'hf'/'hub'/'models--fixture--encoder'
    snapshot = hub/'snapshots'/'testcommit'
    snapshot.mkdir(parents=True)
    (hub/'refs').mkdir()
    (hub/'refs'/'main').write_text('testcommit')
    for name in ['openvino_model.xml','openvino_model.bin','preprocessor_config.json']:
        shutil.copyfile(d/name,snapshot/name)
    env['HF_HOME'] = str(d/'hf')
    p = subprocess.run([str(binary),'--ov-library',library,'-hf','fixture/encoder','--offline','--device','CPU','embed',str(d/'red.png')],env=env,capture_output=True,text=True)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)['embedding'] == [1,0,0]
print('CPU smoke tests passed')
