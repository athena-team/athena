
## Install from the source code

### Creating a virtual environment [Optional]

This project has only been tested on Python 3. We highly recommend creating a virtual environment and installing the python requirements there.

```bash
# Setting up virtual environment
python -m venv venv_athena
source venv_athena/bin/activate
```

### Install *tensorflow* backend

For more information, you can checkout the [tensorflow website](https://github.com/tensorflow/tensorflow).

```bash
# we highly recommend firstly update pip
pip install --upgrade pip
pip install tensorflow==2.0.0
```

### Install *horovod* for multiple-device training [Optional]

For multiple GPU/CPU training
You have to install the *horovod*, you can find out more information from the [horovod website](https://github.com/horovod/horovod#install).

### Install *pydecoder* for WFST decoding [Optional]

For WFST decoding
You have to install *pydecoder*, installation guide for *pydecoder* can be found [athena-decoder website](https://github.com/athena-team/athena-decoder#installation)

### Install *athena* package

```bash
git clone https://github.com/athena-team/athena.git
cd athena
pip install -r requirements.txt
python setup.py bdist_wheel sdist
python -m pip install --ignore-installed dist/athena-0.1.0*.whl
```

- Once athena is successfully installed , you should do `source tools/env.sh` firstly before doing other things.
- For installing some other supporting tools, you can check the `tools/install*.sh` to install kenlm, sph2pipe, spm and ... [Optional]

### Test your installation

- On a single cpu/gpu

```bash
source tools/env.sh
python examples/translate/spa-eng-example/prepare_data.py examples/translate/spa-eng-example/data/train.csv
python athena/main.py examples/translate/spa-eng-example/transformer.json
```

- On multiple cpu/gpu in one machine (you should make sure your hovorod is successfully installed)

```bash
source tools/env.sh
python examples/translate/spa-eng-example/prepare_data.py examples/translate/spa-eng-example/data/train.csv
horovodrun -np 4 -H localhost:4 athena/horovod_main.py examples/translate/spa-eng-example/transformer.json
```

### Notes

- If you see errors such as `ERROR: Cannot uninstall 'wrapt'` while installing TensorFlow, try updating it using command `conda update wrapt`. Same for similar dependencies such as `entrypoints`, `llvmlite` and so on.
- You may want to make sure you have `g++` version 7 or above to make sure you can successfully install TensorFlow.
