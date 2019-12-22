
# Examples for HKUST

## 1 Transformer

```bash
source env.sh
python examples/asr/hkust/prepare_data.py /tmp-data/dataset/opensource/hkust
python athena/main.py examples/asr/hkust/transformer.json
```

## 2 MTL_Transformer_CTC

```bash
source env.sh
python examples/asr/hkust/prepare_data.py /tmp-data/dataset/opensource/hkust
python athena/main.py examples/asr/hkust/mtl_transformer.json
```
