
# Examples for translate

## 1 Transformer

```bash
source tools/env.sh
python examples/translate/spa-eng-example/prepare_data.py examples/translate/spa-eng-example/data/train.csv
python athena/main.py examples/translate/spa-eng-example/transformer.json
```
