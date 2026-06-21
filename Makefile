# RSI — raccourcis. La cible la plus simple : `make install`.

.PHONY: install build test demo connect clean

## install : compile et connecte RSI à ton agent IA (openclaw, hermes-agent…)
install:
	@./install.sh

## build : compile tous les binaires en release
build:
	cargo build --release --bins

## test : lance toute la suite de tests
test:
	cargo test

## demo : lance la simulation de démonstration
demo:
	cargo run --release --bin rsi-demo

## connect : (re)connecte le serveur MCP aux agents (sans recompiler)
connect:
	cargo run --release --bin rsi-connect

## clean : nettoie les artefacts de build
clean:
	cargo clean
	rm -f forge_checkpoint.json
