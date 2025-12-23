# Makefile for Passive Radar Motion Detector

PYTHON = python
PIP = pip
APP = radar.py
DF_APP = df.py

.PHONY: install run run-dual clean help

help:
	@echo "Usage:"
	@echo "  make install    Install dependencies"
	@echo "  make run        Run the standard radar detector"
	@echo "  make run-dual   Run the dual-frequency analyzer"
	@echo "  make clean      Remove temporary files"

install:
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) $(APP)

run-dual:
	$(PYTHON) $(DF_APP)

clean:
	rm -rf __pycache__
	rm -rf *.egg-info
	rm -rf build
	rm -rf dist
