#!/bin/sh
# Remove sol-hermes and its data
set -e

PLUGIN_DIR="${HERMES_HOME:-$HOME/.hermes}/plugins/sol-hermes"
DATA_DIR="${HERMES_HOME:-$HOME/.hermes}/sol-hermes"

echo "Removing plugin: $PLUGIN_DIR"
rm -rf "$PLUGIN_DIR"

echo "Removing data: $DATA_DIR"
rm -rf "$DATA_DIR"

echo "Done. Remove 'sol-hermes' from plugins.enabled in config.yaml if you added it."
