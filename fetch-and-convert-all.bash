#!/bin/bash
set -exuo pipefail

if [ "${1:-}" != "iamamakefile" ]; then
    echo "Please run from the Makefile" >&2
    exit 1
fi

to_valid_resolution() {
    res="$1"
    stride="$2"
    printf 'scale=0; ((%d / %d) * %d) + 1\n' "$res" "$stride" "$stride" | bc -q
}

cd "$(dirname "$(readlink -f "$0")")"

mkdir -p build/json build/tf out

# get-model writes to ., so work in a temporary directory.
tmp="$(mktemp -d)"
trap "rm -rf '$tmp'" EXIT
GET_MODEL="$(readlink -f deps/simple_bodypix_python/get-model.sh)"
BUILD_JSON="$(readlink -f build/json)"

# process all combinations.
while read model; do
    while read width height; do
        while read output; do
            for pu in cpu gpu; do
                # fix resolution
                stride="$(printf "%s" "$model" | sed -re 's/.*stride([0-9]+).*/\1/')"
                width="$(to_valid_resolution "$width" "$stride")"
                height="$(to_valid_resolution "$height" "$stride")"

                if [ "$width" -ne "$height" ]; then
                    echo "Model requires rectangular input, got ${width}x${height}px (after fixing)" >&2
                    exit 1
                fi

                # download tfjs model.
                model_name="$(printf "%s" "$model" | sed -e 's!bodypix/!!; s!model-!!; s!/!_!g;')"
                json_dir="$BUILD_JSON/$model_name"
                if [ ! -e "$json_dir/stamp" ]; then
                    echo "Downloading model $model_name..."
                    rm -rf "$json_dir"
                    ( cd "$tmp" && "$GET_MODEL" "$model" && mv -T * "$BUILD_JSON/$model_name" )
                    touch "$json_dir/stamp"
                fi

                # convert to tf_frozen_model.
                tf_filename="build/tf/${model_name}.tf"
                if [ ! -e "$tf_filename" ]; then
                    echo "Converting $model_name to tf_frozen_model..."
                    build/pyenv/bin/tfjs_graph_converter --output_format tf_frozen_model \
                        "$json_dir" "$tf_filename".tmp
                    mv "$tf_filename".tmp "$tf_filename"
                fi

                # convert to .tflite with given dimensions and output
                full_model_name="${model_name}_${width}_${height}_${output}_${pu}"
                tflite_fn="out/${full_model_name}.tflite"
                if [ ! -e "$tflite_fn" ]; then
                    echo "Generating $full_model_name..."
                    PATH=build/pyenv/bin:$PATH python3 convert.py "$json_dir/model.json" "$tf_filename" \
                        "$width" "$height" "$pu" "$output" "$tflite_fn".tmp
                    mv "$tflite_fn".tmp "$tflite_fn"
                fi
            done # pu
        done < outputs.txt
    done < sizes.txt
done < models.txt

