"""Verify a prepared pool and both base-calibration budgets before GPU training."""
import hashlib
import json
from pathlib import Path
import numpy as np
from upwash.builder import build_dataset
from upwash.config import load_run_config


def main():
    root=Path('/work/asciibase64/fineweb-edu-llama3-upwash')
    raw=(root/'manifest.json').read_bytes()
    manifest=json.loads(raw)
    assert manifest['source']['revision']=='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
    assert manifest['tokenizer']['revision']=='4e20de362430cd3b72f300e6b0f18e50e7166e08'
    assert manifest['dev_set']['revision']=='2484afe000a6fdf442ba2396391a3bec12644903'
    assert manifest['seed']==0 and manifest['total_tokens']>=205_000_000_000
    paths=sorted((root/'shards').glob('*.bin'))
    assert len(paths)==len(manifest['shards'])
    assert not list((root/'shards').glob('*.tmp'))
    total=0
    for meta in manifest['shards']:
        path=root/'shards'/f"{meta['index']:05d}.bin"
        assert path.stat().st_size==4*meta['num_tokens'],path
        assert json.loads(path.with_suffix('.json').read_text())==meta
        total+=meta['num_tokens']
    assert total==manifest['total_tokens']
    for index in np.random.default_rng(0).choice(len(paths),size=min(5,len(paths)),replace=False):
        tokens=np.memmap(paths[index],dtype='<u4',mode='r')
        assert tokens[0]==128000 and tokens[-1]==128001
        assert tokens.max()<=128255
        assert np.count_nonzero(tokens==128000)==np.count_nonzero(tokens==128001)
    dev=Path('/work/asciibase64/fineweb-edu-dev-tokens/data/validation-00000-of-00001.bin')
    tokens=np.memmap(dev,dtype='<u4',mode='r')
    assert len(tokens)==5360078
    assert np.count_nonzero(tokens==128000)==np.count_nonzero(tokens==128001)==5000
    for path in sorted(Path('configs/base150m').glob('base150m-30b*.json')):
        cfg=load_run_config(path)
        dataset=build_dataset(cfg.train_data)
        assert len(dataset)==228881*32
        x,y=dataset[0]
        assert x.shape==(4096,) and bool((x[1:]==y[:-1]).all())
        assert len(build_dataset(cfg.eval_data))==1308
    result={'manifest_sha256':hashlib.sha256(raw).hexdigest(),'total_tokens':total,
            'num_shards':len(paths),'num_dev_documents':5000,'excluded_documents':manifest['dev_set']['excluded']}
    dest=Path('/work/asciibase64/upwash/validation');dest.mkdir(parents=True,exist_ok=True)
    (dest/'data.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DATA_VALIDATION_OK',json.dumps(result))

if __name__=='__main__':
    main()
