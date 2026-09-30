.PHONY: test bench vet model

UPSTREAM     := https://github.com/Xiaobin-Rong/gtcrn
UPSTREAM_REV := 502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73
VENDOR       := tools/export/vendor

test:
	go test ./...

vet:
	go vet ./...

bench:
	go test -run xxx -bench . -benchtime 3s -cpu 1,$$(nproc) .

# Regenerate weights.bin and testdata/ from upstream. Needs python3 with torch,
# numpy, scipy and soundfile.
model:
	[ -d $(VENDOR) ] || git clone -q $(UPSTREAM) $(VENDOR)
	git -C $(VENDOR) checkout -q $(UPSTREAM_REV)
	python3 tools/export/export.py $(VENDOR)
	sha256sum weights.bin
