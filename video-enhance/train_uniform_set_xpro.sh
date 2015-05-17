#!/bin/bash

TOOLS=build/tools

echo "training."

${TOOLS}/caffe train --solver=models/dl-image-enhance-2-layer-192/solver.prototxt

echo "Done."
