
export PYTHONPATH := $(shell pwd)/wire:$(PYTHONPATH)

train:
	rm -rf checkpoints/*
	python wire/main.py
