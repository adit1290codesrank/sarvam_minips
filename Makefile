PY ?= python
CLIPS := $(patsubst clips/raw/%,%,$(basename $(wildcard clips/raw/clip*)))

.PHONY: env convert align translate gold sheet waitk results test

env:            ## CUDA 12.8 wheels (RTX 50xx needs them); use .../whl/cpu on a machine without a GPU
	uv venv --python 3.12 .venv
	uv pip install --python .venv --index-url https://download.pytorch.org/whl/cu128 torch==2.11.* torchaudio==2.11.*
	uv pip install --python .venv -r requirements.txt
	.venv/bin/python -c "from torchaudio.pipelines import MMS_FA as B; B.get_model(); print('MMS_FA ok')"

convert:        ## clips/raw/* -> clips/24k, clips/16k
	$(PY) tools/convert.py clips/raw/clip*

align:
	mkdir -p align/out
	for c in $(CLIPS); do $(PY) align/align_mms.py clips/16k/$$c.wav transcripts/L1/$$c.txt align/out/$$c; done

translate:
	$(PY) translate/translate.py

gold:           ## drafts only; edit checker/gold/*.json by hand afterwards
	$(PY) checker/eval_checker.py draft-gold

sheet:
	$(PY) checker/eval_checker.py sheet

waitk:
	$(PY) partE_ml/wait_k.py

test:
	$(PY) tests/test_checker.py
	$(PY) prototypes/test_hindi_number_parser.py
	$(PY) partE_ml/wait_k.py --test

results: test convert align translate sheet waitk
	$(PY) tools/make_results.py
