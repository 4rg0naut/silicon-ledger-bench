TB := /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
SDK := /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
XDEV_BIN := /Applications/Xcode.app/Contents/Developer/usr/bin
VENV := tools/.venv/bin/python
SWIFT := $(TB)/swift

export SDKROOT := $(SDK)

.PHONY: build probe memory gemm ane spike coreai coreml all clean venv stability

build:
	$(SWIFT) build -c release

venv: tools/.venv/bin/python

tools/.venv/bin/python: tools/requirements.txt
	python3.11 -m venv tools/.venv
	tools/.venv/bin/pip install -r tools/requirements.txt

probe: build
	.build/release/bench probe

memory: build
	.build/release/bench memory

gemm: build
	.build/release/bench gemm

spike: build
	.build/release/bench spike

ane: build
	.build/release/bench ane

models: models/coreai/.generated

models/coreai/.generated: tools/gen_models.py
	$(VENV) tools/gen_models.py
	touch $@

coreai: build models
	.build/release/bench coreai

coreml: build
	.build/release/bench coreml

all: build
	.build/release/bench all

stability: build
	./tools/stability_run.sh mem_official

clean:
	rm -rf .build
