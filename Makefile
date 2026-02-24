BINARY ?= scriby
VERSION ?= dev
DIST_DIR ?= dist
GOOS_ARCHES ?= darwin/amd64 darwin/arm64 linux/amd64 linux/arm64 windows/amd64

.PHONY: build test clean dist runtime-assets

build:
	go build -o $(BINARY) .

test:
	go test ./...

clean:
	rm -rf $(DIST_DIR)

# Cross-platform CLI bundles under dist/scriby
# - macOS/Linux: .tar.gz
# - Windows: .zip

dist: clean
	mkdir -p $(DIST_DIR)/scriby
	set -e; \
	for target in $(GOOS_ARCHES); do \
		os=$${target%/*}; \
		arch=$${target#*/}; \
		ext=""; \
		if [ "$$os" = "windows" ]; then ext=".exe"; fi; \
		stage="$(DIST_DIR)/scriby/$(BINARY)-$(VERSION)-$$os-$$arch"; \
		mkdir -p "$$stage"; \
		GOOS=$$os GOARCH=$$arch CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o "$$stage/$(BINARY)$$ext" .; \
		if [ "$$os" = "windows" ]; then \
			( cd "$$stage" && zip -q -r "../$(BINARY)-$(VERSION)-$$os-$$arch.zip" . ); \
		else \
			( cd "$$stage" && tar -czf "../$(BINARY)-$(VERSION)-$$os-$$arch.tar.gz" . ); \
		fi; \
		rm -rf "$$stage"; \
	done

# Packages whisper-cli runtime binaries from runtime/bin/<os>_<arch>/whisper-cli(.exe)
# and generates dist/runtime/runtime-manifest.json with checksums.
runtime-assets:
	./scripts/package-runtime-assets.sh "$(VERSION)" "$(DIST_DIR)/runtime" "https://github.com/nibzard/scriby/releases/download/$(VERSION)"
