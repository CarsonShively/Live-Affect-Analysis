ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))


VENV := $(ROOT).venv

PY := $(VENV)/bin/python

UV := $(VENV)/bin/uv

HF := $(VENV)/bin/hf

STAMP := $(VENV)/installed

ST := $(VENV)/bin/streamlit

$(PY):
	python3 -m venv $(VENV) 

$(UV): $(PY)
	$(PY) -m pip install uv

lock: $(UV)
	cd $(ROOT) && $(UV) lock


$(STAMP): $(ROOT)uv.lock $(UV)
	cd $(ROOT) && $(UV) sync
	touch $(STAMP) 

install: $(STAMP)

demo: $(STAMP)
	cd $(ROOT) && $(ST) run demo.py --server.fileWatcherType none

labels_eda: $(STAMP)
	cd $(ROOT) && $(PY) labels_eda.py 

process_images: $(STAMP)
	cd $(ROOT) && $(PY) process_images.py 