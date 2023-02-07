export USAGE
.EXPORT_ALL_VARIABLES:
VERSION := $(shell hatch version)
PROJECTNAME := $(shell basename "$(PWD)")
BUILD := $(shell git rev-parse --short HEAD)
DOCKERID = $(shell echo "nuxion")

lock:
	hatch run pip-compile -o requirements.txt  pyproject.toml

lock-transformers:
	hatch run pip-compile --extra transformers -o requirements.transformers.txt  pyproject.toml

debug:
	echo "Not implemented"

deploy:
	echo "Not implemented"

build-local:
	docker build . -t ${DOCKERID}/${PROJECTNAME}
	docker tag ${DOCKERID}/${PROJECTNAME} ${DOCKERID}/${PROJECTNAME}:${VERSION}

build:
	echo "Not implemented"

publish: 
	echo "Not implemented"

.PHONY: docs-serve
docs-serve:
	hatch run sphinx-autobuild docs/source docs/build/html --port 9292 --watch ./
